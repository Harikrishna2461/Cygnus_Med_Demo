"""
Shunt Ligation and Treatment Plan Generator
Uses RAG and LLM to generate personalized ligation strategies.
Incorporates medical knowledge base and shunt-specific pathways.
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LigationPlan:
    """Treatment recommendation for a detected shunt."""
    shunt_type: str
    primary_intervention: str
    secondary_interventions: List[str]
    compression_protocol: str
    follow_up_schedule: str
    contraindications: List[str]
    clinical_rationale: str


class ShuntLigationGenerator:
    """
    Generates LLM-based ligation and treatment plans using RAG context.
    Provides shunt-specific, evidence-based recommendations.
    """
    
    # Shunt-specific CHIVA surgical ligation pathways
    # Focus exclusively on surgical procedures (saphenous-vein-sparing hemodynamic approach)
    TREATMENT_PATHWAYS = {
        "Type 1": {
            "description": "Simple Reflux (N1-N2-N1)",
            "reflux_characteristics": "N1→N2→N1 reflux through GSV",
            "primary_target": "Saphenofemoral junction valve insufficiency",
            "intervention_options": [
                "CHIVA SFJ Ligation with Crossectomy: Groin incision, complete GSV ligation at SFJ with flush ligation of all tributaries at junction, preserve distal GSV if competent",
                "Limited CHIVA: SFJ entry point ligation alone with preservation of distal saphenous trunk for hemodynamic purposes",
                "Selective tributary ligation via small incisions at incompetent tributary origins"
            ],
            "compression_duration_weeks": 6,
            "followup_interval_weeks": 2,
            "has_perforator_involvement": False,
            "chiva_approach": "Hemodynamic assessment - determine if distal GSV compression can be preserved"
        },
        "Type 2": {
            "description": "Reflux with Perforating Involvement (N2-N3)",
            "reflux_characteristics": "N2→N3 tributary reflux",
            "primary_target": "Tributary branches with autonomous drainage",
            "intervention_options": [
                "CHIVA Tributary Ligation: Selective ligation of incompetent tributaries via limited groin and thigh incisions, preserve main GSV if hemodynamically competent",
                "Selective microphlebectomy of superficial tributary branches",
                "Entry point ligation of tributary at SFJ while preserving GSV trunk"
            ],
            "compression_duration_weeks": 4,
            "followup_interval_weeks": 3,
            "has_perforator_involvement": True,
            "chiva_approach": "Preserve saphenous trunk when possible - only ligate incompetent tributaries"
        },
        "Type 3": {
            "description": "Complex Multi-level (N1-N2-N3-N1)",
            "reflux_characteristics": "N1→N2→N3→N1 forming complete loop",
            "primary_target": "GSV trunk with tributary involvement",
            "intervention_options": [
                "CHIVA Staged Approach: Phase 1 - SFJ entry point ligation with maximal tributary preservation; Phase 2 (if needed) - selective perforator and distal tributary ligation",
                "Hemodynamic SFJ ligation with careful preservation of competent tributary drainage pathways",
                "High SFJ ligation with selective perforator ligation below the knee if indicated"
            ],
            "compression_duration_weeks": 10,
            "followup_interval_weeks": 2,
            "has_perforator_involvement": True,
            "chiva_approach": "Staged surgical approach - initial entry point ligation, reassess before secondary procedures"
        },
        "Type 4 Pelvic": {
            "description": "Pelvic Origin Reflux (P-N2-N1)",
            "reflux_characteristics": "Pelvic source (gonadal/iliac) → GSV → deep system",
            "primary_target": "Pelvic source with proximal GSV entry",
            "intervention_options": [
                "CHIVA Pelvic Ligation: High SFJ ligation above gonadal vein entry with preservation of pelvic venous continuity, selective gonadal vein ligation if incompetent",
                "Gonadal vein ligation at groin level with SFJ preservation if possible",
                "High saphenofemoral junction ligation with flush closure of gonadal entry - requires surgical expertise"
            ],
            "compression_duration_weeks": 12,
            "followup_interval_weeks": 2,
            "has_perforator_involvement": False,
            "chiva_approach": "Assess pelvic anatomy carefully - ligate source while preserving collateral venous drainage"
        },
        "Type 4 Perforator": {
            "description": "Perforator Origin (N1-N3-N2-N1)",
            "reflux_characteristics": "Bone perforator origin with secondary saphenous involvement",
            "primary_target": "Incompetent perforator at entry point (fascia level)",
            "intervention_options": [
                "CHIVA Perforator Ligation (SEPS): Subfascial endoscopic approach to ligate incompetent perforator at fascia level via calf incision, preserve saphenous system",
                "Direct subfascial perforator ligation: Open approach for large incompetent perforators",
                "Limited incision perforator ligation at fascia entry point - hemodynamic preservation of tributaries"
            ],
            "compression_duration_weeks": 8,
            "followup_interval_weeks": 2,
            "has_perforator_involvement": True,
            "chiva_approach": "Subfascial ligation approach - interrupt reflux at fascia without removing venous channels"
        },
        "Type 5 Pelvic": {
            "description": "Complex Pelvic Reflux (P-N3-N2-N1)",
            "reflux_characteristics": "Pelvic origin with complex tributary participation",
            "primary_target": "Pelvic source with extensive secondary involvement",
            "intervention_options": [
                "CHIVA Comprehensive Ligation: Staged approach - Phase 1: High SFJ and pelvic source ligation; Phase 2 (if needed): Selective tributary ligation",
                "High SFJ ligation with gonadal vein assessment and selective ligation based on hemodynamic patterns",
                "Complex staged surgical approach with interval assessment (4-6 weeks) before secondary procedures"
            ],
            "compression_duration_weeks": 12,
            "followup_interval_weeks": 2,
            "has_perforator_involvement": True,
            "chiva_approach": "Staged CHIVA with hemodynamic emphasis - determine which vessels require ligation vs. preservation"
        },
        "Type 5 Perforator": {
            "description": "Complex Perforator Reflux (N1-N3-N2-N3-N1)",
            "reflux_characteristics": "Multiple perforator involvement with biphasic reflux",
            "primary_target": "Multiple perforator entry points",
            "intervention_options": [
                "CHIVA Multiple Perforator Ligation: SEPS approach with ligation of all hemodynamically significant perforators identified on ultrasound mapping",
                "Staged perforator ligation: Initial ligation of major incompetent perforators, reassess at 6 weeks for additional ligation",
                "Subfascial approach with selective perforator interruption while preserving saphenous and deep venous systems"
            ],
            "compression_duration_weeks": 10,
            "followup_interval_weeks": 2,
            "has_perforator_involvement": True,
            "chiva_approach": "Multiple subfascial perforator ligation - determine extent via detailed ultrasound hemodynamic assessment"
        },
        "Type 6": {
            "description": "Deep-to-Tributary Reflux (N1-N3-N2)",
            "reflux_characteristics": "Direct deep-to-tributary with possible saphenous involvement",
            "primary_target": "Perforating pathway (deep vein entry)",
            "intervention_options": [
                "CHIVA Perforator Ligation: Targeted subfascial ligation of incompetent perforator(s) at entry point without saphenous intervention",
                "Selective perforator ligation via small incisions at fascial level - calf or medial thigh approach",
                "Ultrasound-guided identification of dominant perforator with limited surgical ligation approach"
            ],
            "compression_duration_weeks": 8,
            "followup_interval_weeks": 3,
            "has_perforator_involvement": True
        }
    }
    
    def __init__(self, llm_callable, retrieval_callable):
        """
        Initialize ligation generator.
        
        Args:
            llm_callable: Function to call LLM with prompt (e.g., call_llm from app.py)
            retrieval_callable: Function to retrieve RAG context (e.g., retrieve_context)
        """
        self.llm_callable = llm_callable
        self.retrieval_callable = retrieval_callable
    
    def generate_treatment_plan(
        self,
        shunt_type: str,
        flow_pattern: List[str],
        patient_context: Dict,
        reasoning: str
    ) -> Dict:
        """
        Generate comprehensive LLM-based treatment plan.
        
        Args:
            shunt_type: Detected shunt type (e.g., "Type 3")
            flow_pattern: Flow sequence (e.g., ["N1", "N2", "N3", "N1"])
            patient_context: Patient-specific data including:
                - age, symptoms, hemodynamic_class, contraindications, etc.
            reasoning: Clinical reasoning from flow analysis
        
        Returns:
            Complete ligation plan with treatments, follow-up, and rationale
        """
        try:
            # Get pathway-specific information
            pathway = self.TREATMENT_PATHWAYS.get(
                shunt_type,
                self.TREATMENT_PATHWAYS["Type 1"]  # Default fallback
            )
            
            # Retrieve RAG context for this shunt type
            rag_query = f"{shunt_type} {pathway['description']} treatment management"
            rag_context = self.retrieval_callable(rag_query) if self.retrieval_callable else []
            medical_context = "\n".join(rag_context) if rag_context else ""
            
            # Build LLM prompt with CHIVA surgical ligation focus
            chiva_approach = pathway.get('chiva_approach', 'Hemodynamic surgical approach')
            
            prompt = f"""You are a vascular surgeon specialist in CHIVA (Cure Hémodynamique de l'Insuffisance Veineuse en Ambulatoire) surgical techniques.
Generate a detailed, personalized SURGICAL LIGATION plan based on hemodynamic principles and saphenous-vein-sparing strategies.
CRITICAL: Focus EXCLUSIVELY on surgical ligation procedures. No endovascular ablation, no sclerotherapy - SURGICAL LIGATION ONLY.

=== DETECTED SHUNT ANALYSIS ===
Shunt Type: {shunt_type}
Pattern: {' → '.join(flow_pattern)}
Description: {pathway['description']}
Reflux: {pathway['reflux_characteristics']}
CHIVA Strategy: {chiva_approach}

Clinical Reasoning: {reasoning}

Patient Context:
{self._format_patient_context(patient_context)}

=== CHIVA SURGICAL PATHWAY OPTIONS (Saphenous-Sparing Hemodynamic Approach) ===
{chr(10).join([f"- {opt}" for opt in pathway['intervention_options']])}

=== MEDICAL KNOWLEDGE BASE ===
{medical_context[:2000]}  # Limit to prevent token overflow

=== REQUIRED OUTPUT FORMAT FOR SURGICAL LIGATION ===
Provide EXACTLY this structure (no numbering, no asterisks, no special chars):

Primary Intervention: [Specific surgical ligation procedure with anatomical details, incision location, vessels to be ligated, hemodynamic consideration]

Secondary Interventions:
[Specific surgical procedure 2 - approach, location, vessels, preservation strategy if applicable]
[Specific surgical procedure 3 - surgical details, timing relative to primary intervention]
[Optional staged procedure - if indicated for complex cases]

Compression Protocol: [Post-operative compression strategy with mmHg level, duration {pathway['compression_duration_weeks']} weeks, application details]

Follow-up Schedule: [Post-operative timeline: wound check (days), duplex ultrasound (weeks), clinical assessment (weeks), return to activity]

Key Contraindications: [Critical surgical contraindications for this patient and shunt type]

Clinical Rationale: [One paragraph explaining the hemodynamic surgical strategy, which vessels are being ligated, which are being preserved, and how this addresses the {' → '.join(flow_pattern)} reflux pattern]

Procedure Notes: [2-3 key intra-operative surgical considerations: positioning, key anatomical landmarks, critical safety measures]

=== CRITICAL RULES FOR CHIVA SURGICAL LIGATION ===
- SURGICAL PROCEDURES ONLY: SFJ ligation, tributary ligation, perforator ligation (SEPS), microphlebectomy
- Match every recommendation to the SPECIFIC shunt type and flow pattern
- Include exact surgical approaches: groin incision for SFJ, calf incision for perforators, limited incisions for tributaries
- Include specific vessels: which to ligate, which to preserve based on hemodynamic assessment
- Mention saphenous preservation where appropriate (CHIVA principle of hemodynamic assessment)
- Duration must match {pathway['compression_duration_weeks']} week standard for {shunt_type}
- Reference the specific flow pattern ({' → '.join(flow_pattern)}) in rationale
- NO generic recommendations
- NO endovascular, NO ablation, NO sclerotherapy
- Interventions are ONE sentence each
- Emphasize hemodynamic reasoning (why this vessel, why at this location)"""
            
            # Call LLM
            response = self.llm_callable(prompt, stream=False)
            
            # Parse response into structured format
            plan = self._parse_ligation_response(response, shunt_type, pathway)
            
            logger.info(f"Generated ligation plan for {shunt_type}")
            return plan
            
        except Exception as e:
            logger.error(f"Error generating ligation plan: {e}")
            return self._create_fallback_plan(shunt_type)
    
    def _format_patient_context(self, patient_context: Dict) -> str:
        """Format patient context for LLM prompt."""
        lines = []
        if patient_context.get("age"):
            lines.append(f"Age: {patient_context['age']} years")
        if patient_context.get("hemodynamic_class"):
            lines.append(f"Hemodynamic Class (C0-C6): {patient_context['hemodynamic_class']}")
        if patient_context.get("symptoms"):
            lines.append(f"Symptoms: {patient_context['symptoms']}")
        if patient_context.get("comorbidities"):
            lines.append(f"Comorbidities: {patient_context['comorbidities']}")
        if patient_context.get("previous_treatment"):
            lines.append(f"Previous Treatment: {patient_context['previous_treatment']}")
        if patient_context.get("contraindications"):
            lines.append(f"Contraindications: {patient_context['contraindications']}")
        
        return "\n".join(lines) if lines else "No specific patient context provided"
    
    def _parse_ligation_response(
        self,
        response: str,
        shunt_type: str,
        pathway: Dict
    ) -> Dict:
        """Parse LLM response into structured plan."""
        plan = {
            "shunt_type": shunt_type,
            "response_full_text": response,
            "primary_intervention": self._extract_section(response, "Primary Intervention:"),
            "secondary_interventions": self._extract_list_section(
                response, "Secondary Interventions:"
            ),
            "compression_protocol": self._extract_section(response, "Compression Protocol:"),
            "follow_up_schedule": self._extract_section(response, "Follow-up Schedule:"),
            "contraindications": self._extract_list_section(
                response, "Key Contraindications:"
            ),
            "clinical_rationale": self._extract_section(response, "Clinical Rationale:"),
            "procedure_notes": self._extract_list_section(response, "Procedure Notes:"),
            "pathway_info": {
                "description": pathway["description"],
                "primary_target": pathway["primary_target"],
                "has_perforator_involvement": pathway.get("has_perforator_involvement", False),
                "requires_ir_evaluation": pathway.get("requires_ir_evaluation", False)
            }
        }
        
        return plan
    
    def _extract_section(self, text: str, section_header: str) -> str:
        """Extract a single section from LLM response."""
        try:
            start_idx = text.find(section_header)
            if start_idx == -1:
                return ""
            
            start_idx += len(section_header)
            
            # Find next section or end of text
            next_sections = [
                "Secondary Interventions:",
                "Compression Protocol:",
                "Follow-up Schedule:",
                "Key Contraindications:",
                "Clinical Rationale:",
                "Procedure Notes:"
            ]
            
            end_idx = len(text)
            for section in next_sections:
                idx = text.find(section, start_idx)
                if idx != -1 and idx < end_idx:
                    end_idx = idx
            
            content = text[start_idx:end_idx].strip()
            return content.lstrip("\n").rstrip()
            
        except Exception as e:
            logger.warning(f"Error extracting section {section_header}: {e}")
            return ""
    
    def _extract_list_section(self, text: str, section_header: str) -> List[str]:
        """Extract a list section from LLM response."""
        content = self._extract_section(text, section_header)
        if not content:
            return []
        
        # Split by newlines and filter empties
        items = [line.strip() for line in content.split("\n") if line.strip()]
        
        # Remove leading dashes/bullets
        items = [
            item.lstrip("-•*").strip() for item in items
        ]
        
        return items
    
    def _create_fallback_plan(self, shunt_type: str) -> Dict:
        """Create fallback plan if LLM generation fails."""
        pathway = self.TREATMENT_PATHWAYS.get(
            shunt_type,
            self.TREATMENT_PATHWAYS["Type 1"]
        )
        
        return {
            "shunt_type": shunt_type,
            "primary_intervention": pathway["intervention_options"][0],
            "secondary_interventions": pathway["intervention_options"][1:3],
            "compression_protocol": (
                f"{pathway['compression_duration_weeks']} weeks "
                f"compression with 20-30mmHg graduated compression stockings"
            ),
            "follow_up_schedule": (
                f"DSA/DUS at {pathway['followup_interval_weeks']} weeks, "
                f"then monthly for 3 months"
            ),
            "contraindications": [
                "Acute DVT in target vein",
                "Severe arterial insufficiency",
                "Active infection"
            ],
            "clinical_rationale": pathway['description'],
            "is_fallback": True
        }
    
    def generate_quick_ligation_summary(self, shunt_type: str) -> str:
        """Generate quick reference summary for shunt type."""
        pathway = self.TREATMENT_PATHWAYS.get(shunt_type)
        if not pathway:
            return f"Unknown shunt type: {shunt_type}"
        
        return f"""
{shunt_type}: {pathway['description']}

Primary Target: {pathway['primary_target']}

Main Options: {', '.join([opt.split(':')[0] for opt in pathway['intervention_options'][:3]])}

Compression: {pathway['compression_duration_weeks']} weeks standard

Follow-up: Every {pathway['followup_interval_weeks']} weeks
""".strip()


# Example usage function (can be called from app.py)
def create_ligation_generator(call_llm_func, retrieve_context_func):
    """Factory function to create generator with proper dependencies."""
    return ShuntLigationGenerator(call_llm_func, retrieve_context_func)
