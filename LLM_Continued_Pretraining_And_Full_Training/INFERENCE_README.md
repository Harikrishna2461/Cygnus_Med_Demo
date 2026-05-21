# Medical LLM Inference - Test & Evaluation Scripts

After training the model with `pretrain.ipynb`, use these scripts to evaluate its performance on medical shunt classification tasks.

## Scripts Overview

### 1. **test_inference_quick.py** (Recommended for quick testing)
Fast inference test on 5 representative shunt classification cases.

**Use when:** You want rapid feedback on model output
```bash
python test_inference_quick.py
```

**Output:**
- Console: Real-time classification results for each test case
- `logs/quick_inference_results.json`: Structured results with classifications

**Expected Output Structure:**
```json
{
  "classification": "Type 1",
  "ligation_strategy": "Ligate at N2->N1 junction"
}
```

### 2. **test_inference.py** (Comprehensive testing)
Full inference test suite with trained model vs. base model comparison.

**Use when:** You need detailed comparison or testing more cases
```bash
python test_inference.py
```

**Output:**
- Console: Test-by-test results with JSON validation
- `logs/inference_results.json`: Complete results with all metadata

### 3. **analyze_model_output.py** (Diagnostic)
Analyzes what the model is actually learning - medical knowledge or template copying?

**Use when:** Model outputs don't look right and you need to diagnose the issue
```bash
python analyze_model_output.py
```

**Output:**
- Console: Raw model responses to diagnostic prompts
- `logs/model_analysis.json`: Saved analysis for review

## Key Improvements in Latest Version

### Better Prompts
- No literal answer templates that encourage copying
- Explicit medical classifications defined in prompt
- Clear JSON format instructions
- CHIVA rules embedded for domain knowledge

### Optimized Generation
- `do_sample=False`: Greedy decoding for consistent output
- `temperature=0.3`: Lower temperature for deterministic JSON
- `max_new_tokens=150`: Enough for complete JSON response
- Automatic JSON extraction from model output

### Better Output Handling
- Automatic JSON extraction even if model adds extra text
- Clear display of classifications and ligation strategies
- Structured JSON storage for analysis

## Expected Behavior

### Model Should:
✓ Output valid JSON with `shunt_classification` and `ligation` keys
✓ Classify as: Type 1, Type 2A, Type 2B, Type 2C, Type 1+2, or No Shunt
✓ Provide appropriate ligation strategy (e.g., "Ligate N2->N1")

### Model Should NOT:
✗ Copy literal text from prompt
✗ Output malformed JSON
✗ Hallucinate medical terms not in training data
✗ Ignore the clip data input

## Troubleshooting

### Issue: Low accuracy / wrong classifications
**Solution:**
1. Run `analyze_model_output.py` to see what model learned
2. Check if model is generating medical knowledge or just repeating template
3. May need more training epochs or better data quality

### Issue: Invalid JSON output
**Solution:**
1. Check `logs/inference_results.json` for raw responses
2. Look for extra text before/after JSON (script auto-extracts core JSON)
3. If format is still invalid, review prompt structure

### Issue: Very slow inference
**Solution:**
1. Reduce `max_new_tokens` in generation config
2. Use `do_sample=False` (greedy decoding is faster)
3. Ensure GPU memory is available: `nvidia-smi`

## Running on Ubuntu

### Setup
```bash
# Ensure trained model is available
ls medical_qwen_cpt/

# Ensure logs directory exists
mkdir -p logs

# Check GPU
nvidia-smi
```

### Quick Test (recommended first step)
```bash
python test_inference_quick.py
```

### Full Test Suite
```bash
python test_inference.py
```

### Diagnostic Analysis
```bash
python analyze_model_output.py
```

## Understanding Results

### JSON Output Format
```json
{
  "shunt_classification": "Type 1",
  "ligation_strategy": "Ligate at EP-RP junction"
}
```

### Accuracy Metrics
- **Classification Accuracy**: % of correct shunt type predictions
- **JSON Validity**: % of valid JSON responses
- **Task Completion**: % of tests with parseable output

## Files Generated

| File | Purpose | Review When |
|------|---------|-------------|
| `logs/quick_inference_results.json` | Quick test results | After quick test |
| `logs/inference_results.json` | Full test suite results | After comprehensive test |
| `logs/model_analysis.json` | Diagnostic outputs | Troubleshooting |

## Medical Context

### CHIVA Shunt Types
- **Type 1**: Superficial truncal incompetence with direct deep-to-superficial reflux
- **Type 2A**: Perforator incompetence + superficial venous incompetence
- **Type 2B**: Perforator incompetence with superficial reflux into deep
- **Type 2C**: Perforator incompetence draining into deep system
- **Type 1+2**: Combined Type 1 and Type 2 shunt
- **No Shunt**: Normal hemodynamics

### Clip Data Interpretation
- `EP at N1->N2`: Entry point at node 1 flowing to node 2
- `RP at N2->N1`: Re-entry point at node 2 flowing back to node 1
- Multiple clips indicate complex multi-node shunts

---

**Status**: Ready for Ubuntu testing ✓
