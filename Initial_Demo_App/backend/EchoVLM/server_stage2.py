import os
import asyncio
import json
import base64
import time
import uuid
from typing import List, Optional, Any, Dict, Union, Tuple
from datetime import datetime
from io import BytesIO
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from PIL import Image
import openai
import imghdr
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 请求模型【核心修改1：新增doctor_revise字段】 ====================
class InferenceRequest(BaseModel):
    doctor_id: str
    case_id: str
    image_numbers: Optional[List[str]] = Field(default_factory=list)
    images: List[str] = Field(default_factory=list)
    query_text: str
    # 新增：医生修正内容，非必填，为空则使用AI原始回答作为上下文，有值则用该值替换AI回答存入上下文
    doctor_revise: Optional[str] = Field(default=None)


# ==================== 整合服务类 ====================
class InferenceService:
    def __init__(self, openai_api_key, openai_base_url, openai_model, api_timeout, temp_image_dir,
                 conversation_save_dir, save_image_json):
        # 把配置参数从硬编码改为初始化参数，更灵活
        self.client = openai.AsyncOpenAI(api_key=openai_api_key, base_url=openai_base_url)
        self.temp_dir = Path(temp_image_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir = Path(conversation_save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_image_json = save_image_json
        self.openai_model = openai_model
        self.api_timeout = api_timeout

    def _load_conversation_history(self, doctor_id: str, case_id: str) -> List[Dict[str, str]]:
        """加载历史对话记录【核心修改2：优先读取医生修正内容作为assistant的回复】
        返回格式：[{"role": "user/assistant", "content": "文本内容"}, ...]
        核心逻辑：历史轮次中，如果有doctor_revise就用修正值，无则用AI原始回答
        """
        doctor_dir = self.save_dir / doctor_id
        file_path = doctor_dir / f"{case_id}.json"

        history_messages = []
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # 遍历历史轮次，转换为OpenAI的messages格式
                for turn in data.get("turns", []):
                    # 追加用户提问（仅文本，图片只在本轮传递）
                    history_messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": turn["query_text"]}]
                    })
                    # 【核心变更】优先使用医生修正的内容，没有则用AI原始回复
                    assistant_content = turn.get("doctor_revise") or turn["ai_response"]
                    history_messages.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": assistant_content}]
                    })
            except Exception as e:
                logger.warning(f"加载历史对话失败，将使用空上下文: {str(e)}")
        return history_messages

    async def run_inference(
            self,
            doctor_id: str,
            case_id: str,
            query_text: str,
            images: List[str],
            doctor_revise: Optional[str] = None,  # 【核心修改3：新增修正内容入参】
            n_results: int = 1
    ) -> Dict[str, Any]:
        # 1. 处理图片（生成base64和mime类型）
        image_payloads = await self._process_images_multi_type(images)

        # 2. 加载历史对话上下文（已做修正内容优先加载）
        history_messages = self._load_conversation_history(doctor_id, case_id)

        # 3. 构建本轮用户消息（文本+图片）
        current_user_content = [{"type": "text", "text": query_text}]
        for b64, mime, _ in image_payloads:
            current_user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"}
            })

        # 4. 拼接完整的对话上下文：历史 + 本轮用户消息
        full_messages = history_messages + [{
            "role": "user",
            "content": current_user_content
        }]

        # 5. 调用OpenAI API（传入完整上下文）
        start = time.time()
        responses = await self._call_openai(full_messages, n_results)
        cost_time = time.time() - start

        # 6. 保存本轮对话（追加到历史，传入医生修正内容）
        if self.save_image_json:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._save_conversation,
                doctor_id,
                case_id,
                query_text,
                responses[0],
                doctor_revise,  # 【核心修改4：传入医生修正内容到保存方法】
                [p for _, _, p in image_payloads]
            )

        return {
            "doctor_id": doctor_id,
            "case_id": case_id,
            "ai_response": responses[0],
            "doctor_revise": doctor_revise, # 返回医生传入的修正内容
            "cost_time": cost_time,
            "history_turns_count": len(history_messages) // 2  # 历史轮次数量（每2条为一轮）
        }

    # ---------------- 图片处理 无修改 ----------------
    async def _process_images_multi_type(self, image_inputs: List[str]):
        tasks = [self._process_single_image(idx, img) for idx, img in enumerate(image_inputs)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for idx, r in enumerate(results):
            if isinstance(r, Exception):
                raise HTTPException(status_code=400, detail=f"图片处理失败 #{idx}: {str(r)}")
        return results

    @staticmethod
    def _get_image_mime_type(image_path: str) -> str:
        ext = os.path.splitext(image_path)[1].lower()
        return {
            '.png': 'image/png',
            '.bmp': 'image/bmp',
            '.tiff': 'image/tiff',
            '.tif': 'image/tiff',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg'
        }.get(ext, 'image/jpeg')

    def _get_mime_from_raw(self, raw: bytes) -> str:
        """从字节流判断图片MIME类型"""
        img_type = imghdr.what(None, h=raw)
        logger.info(f"图片类型检测结果: {img_type}")
        if img_type == "jpeg":
            return "image/jpeg"
        elif img_type == "png":
            return "image/png"
        elif img_type == "gif":
            return "image/gif"
        elif img_type == "webp":
            return "image/webp"
        return "image/jpeg"  # 兜底

    @staticmethod
    async def _encode_image_async(image_path: str) -> str:
        loop = asyncio.get_event_loop()
        with open(image_path, "rb") as f:
            data = await loop.run_in_executor(None, f.read)
        return base64.b64encode(data).decode()

    async def _process_single_image(self, idx: int, image_input_or_base_64: str) -> Tuple[str, str, str]:
        """返回 (b64_str, mime, disk_path_for_save)"""
        # 1. 本地文件路径分支：直接用原路径，不落盘
        if len(image_input_or_base_64) < 300 and os.path.isfile(image_input_or_base_64):
            mime = self._get_image_mime_type(image_input_or_base_64)
            b64_str = await self._encode_image_async(image_input_or_base_64)
            return b64_str, mime, image_input_or_base_64

        # 2. Base64字符串分支（带前缀/纯Base64）
        raw: bytes = b""
        save_path = ""
        if "," in image_input_or_base_64:
            # 带前缀的Base64：直接提取body作为Base64字符串，解析前缀里的MIME
            head, b64_str = image_input_or_base_64.split(",", 1)
            mime = head.split(":")[1].split(";")[0]
            # 验证Base64合法性并解码（仅为了后续落盘，不重复编码）
            if self.save_image_json:
                raw = base64.b64decode(b64_str)
        else:
            # 纯Base64：直接用原字符串，从字节流判断MIME
            b64_str = image_input_or_base_64
            raw = base64.b64decode(b64_str)
            mime = self._get_mime_from_raw(raw)

        # 落盘（仅为留路径）：使用已解码的raw字节流，无需重复编解码
        if self.save_image_json and raw:
            save_path = str(
                self.temp_dir /
                f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}_{idx}.jpg"
            )
            await asyncio.get_event_loop().run_in_executor(
                None, Path(save_path).write_bytes, raw
            )

        return b64_str, mime, save_path

    # ---------------- OpenAI调用 无修改 ----------------
    async def _call_openai(self, messages: List[Dict[str, Any]], n_results: int) -> List[str]:
        """调用OpenAI API（适配多轮上下文）"""
        try:
            logger.info("开始调用OpenAI API（多轮模式）...")
            response = await self.client.chat.completions.create(
                model=self.openai_model,
                messages=messages,  # 使用完整的多轮上下文
                n=n_results,
                timeout=self.api_timeout,
                temperature=0.3,
            )
            logger.info("OpenAI API 调用成功")
            logger.info(f"回复结果: {response.choices}")
            return [choice.message.content for choice in response.choices]

        except openai.APITimeoutError:
            logger.error("OpenAI API 超时")
            raise HTTPException(status_code=504, detail="OpenAI API 超时")

        except openai.APIError as e:
            logger.error(f"OpenAI API 错误: {str(e)}")
            raise HTTPException(status_code=500, detail=f"OpenAI API 错误: {str(e)}")

        except Exception as e:
            logger.error(f"未知错误: {str(e)}")
            logger.exception(e)
            raise HTTPException(status_code=500, detail=f"内部错误: {str(e)}")

    def _save_conversation(
            self,
            doctor_id: str,
            case_id: str,
            query_text: str,
            ai_response: str,
            doctor_revise: Optional[str],  # 【核心修改5：新增修正内容入参，持久化存储】
            image_paths: List[str]
    ):
        """保存对话（按doctor_id/case_id.json格式，追加轮次）【新增存储医生修正内容】"""
        # 文件路径：conversations/{doctor_id}/{case_id}.json
        doctor_dir = self.save_dir / doctor_id
        doctor_dir.mkdir(exist_ok=True)
        file_path = doctor_dir / f"{case_id}.json"

        # 加载或创建数据结构
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except:
                data = {"doctor_id": doctor_id, "case_id": case_id, "turns": []}
        else:
            data = {"doctor_id": doctor_id, "case_id": case_id, "turns": []}

        # 追加新的对话轮次【核心修改6：存入医生修正内容+AI原始回答】
        data["turns"].append({
            "query_text": query_text,
            "ai_response": ai_response,       # 保留AI原始回答
            "doctor_revise": doctor_revise,   # 存储医生修正内容
            "image_paths": image_paths,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        # 保存回文件
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"保存对话失败: {str(e)}")


# ==================== FastAPI 实例 ====================
app = FastAPI(title="医学图像VLM分析API（多轮对话+医生修正版）", version="2.1.0")

# 配置参数（集中管理，便于修改）
CONFIG = {
    "OPENAI_API_KEY": "EMPTY",
    "OPENAI_BASE_URL": "http://10.116.39.70:7008/v1",
    "OPENAI_MODEL": '/home/user02/SCY/SonoVLM_V2/checkpoints/ablation/ablation_data/data_scaling/checkpoint-1170-merged',
    "API_TIMEOUT": 60,
    "TEMP_IMAGE_DIR": "/home/user02/SCY/SonoVLM_V2/vllm/images",
    "CONVERSATION_SAVE_DIR": "/home/user02/SCY/SonoVLM_V2/vllm",
    "SAVE_IMAGE_JSON": False
}

# 初始化服务（传入配置）
service = InferenceService(
    openai_api_key=CONFIG["OPENAI_API_KEY"],
    openai_base_url=CONFIG["OPENAI_BASE_URL"],
    openai_model=CONFIG["OPENAI_MODEL"],
    api_timeout=CONFIG["API_TIMEOUT"],
    temp_image_dir=CONFIG["TEMP_IMAGE_DIR"],
    conversation_save_dir=CONFIG["CONVERSATION_SAVE_DIR"],
    save_image_json=CONFIG["SAVE_IMAGE_JSON"]
)


@app.post("/infer", response_model=Dict[str, Any])
async def infer(request: InferenceRequest):
    """多轮推理接口（基于doctor_id+case_id拼接历史上下文+医生修正）【核心修改7：传入修正内容】"""
    try:
        result = await service.run_inference(
            doctor_id=request.doctor_id,
            case_id=request.case_id,
            query_text=request.query_text,
            images=request.images,
            doctor_revise=request.doctor_revise,  # 传递医生修正内容
            n_results=1
        )
        return {"status": "success", "data": result}
    except HTTPException as e:
        return {"status": "failed", "data": None, "detail": e.detail}
    except Exception as e:
        return {"status": "failed", "data": None, "detail": f"内部错误: {str(e)}"}


if __name__ == "__main__":
    uvicorn.run("server_stage2:app", host="10.116.39.70", port=7000, reload=False)