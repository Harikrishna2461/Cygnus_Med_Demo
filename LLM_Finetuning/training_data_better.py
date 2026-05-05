"""
IMPROVED TRAINING DATA - EXPLICIT CHIVA DECISION LOGIC
Focus on teaching classification rules, not just listing facts
"""

def generate_better_training_pairs():
    """
    Create training pairs that explicitly teach the CHIVA classification decision tree
    """

    pairs = [
        # TYPE 1 - SFJ INCOMPETENT (EP N1→N2), GSV REFLUX, NO OTHER PATHWAYS
        {
            "instruction": "Classify this venous shunt: SFJ is INCOMPETENT (reflux detected at EP N1→N2). GSV shows reflux distally. No perforator involvement. No DV-N3 connections. What type?",
            "response": """DECISION TREE:
Step 1: Check for EP N1→N2 (SFJ/Hunterian entry) → YES, INCOMPETENT
Step 2: Is this DV→N2 entry (Hunterian)? → NO, it's SFJ
Step 3: Any perforators feeding the reflux? → NO
Step 4: Any DV-N3 (tributary) connections? → NO

CLASSIFICATION: TYPE 1 (SFJ-Based Shunt)
REASONING: SFJ is incompetent with GSV reflux. No complex pathways.
LIGATION: Saphenofemoral junction ONLY. Disconnect N1→N2 flow."""
        },

        # TYPE 2A - SFJ COMPETENT, GSV→N3 (tributary) CONNECTION
        {
            "instruction": "Classify: SFJ is COMPETENT (no EP N1→N2). GSV connects to a tributary at mid-thigh (EP N2→N3). Tributary shows reflux. No DV involvement. What type?",
            "response": """DECISION TREE:
Step 1: Check for EP N1→N2 → NO, SFJ is COMPETENT
Step 2: Check for EP N2→N3 (GSV to tributary) → YES, present
Step 3: Is N3 connected to DV (N1)? → NO, only connected to GSV
Step 4: Identify the feeding path → GSV feeds tributary directly

CLASSIFICATION: TYPE 2A (GSV-Tributary Connection)
REASONING: SFJ competent but GSV has pathological entry into tributary.
LIGATION: Ligate GSV at the point where it connects to the tributary.
Principle: Preserve GSV function while blocking tributary entry."""
        },

        # TYPE 2B - SFJ COMPETENT, DV→N3 (perforator to tributary)
        {
            "instruction": "Classify: SFJ is COMPETENT. Deep femoral vein connects directly to a tributary (EP N1→N3, perforator). No GSV involvement in this connection. What type?",
            "response": """DECISION TREE:
Step 1: Check for EP N1→N2 → NO
Step 2: Check for GSV→N3 connection → NO, GSV not involved
Step 3: Check for DV→N3 (perforator-tributary) → YES
Step 4: Confirm no additional pathways → Only DV→N3 present

CLASSIFICATION: TYPE 2B (DV-Perforator-Tributary)
REASONING: Deep system feeds tributary directly via perforator.
LIGATION: Ligate the perforator at its DV entry and at the tributary.
Principle: Block both ends of the abnormal perforator to prevent drainage."""
        },

        # TYPE 2C - SFJ COMPETENT, ALL THREE SYSTEMS INVOLVED
        {
            "instruction": "Classify: SFJ COMPETENT. GSV connects to DV (perforator, EP N2→N1). That DV also connects to tributaries (N1→N3). Complex three-system involvement. What type?",
            "response": """DECISION TREE:
Step 1: Check for EP N1→N2 → NO, SFJ competent
Step 2: Check for N2→N3 (GSV-tributary direct) → NO
Step 3: Check for N1→N3 (DV-tributary direct) → YES
Step 4: Check for N2→N1 (GSV-DV perforator) → YES
Step 5: Multiple pathways identified → Type 2C

CLASSIFICATION: TYPE 2C (Complex GSV-DV-Tributary)
REASONING: Three systems connected: GSV feeds DV via perforator, DV feeds tributaries.
LIGATION: Ligate BOTH the GSV-DV perforator AND the DV-tributary connections.
Principle: Two separate ligation points needed to fully disconnect the complex circuit."""
        },

        # TYPE 1+2 - SFJ INCOMPETENT PLUS INDEPENDENT TRIBUTARY
        {
            "instruction": "Classify: SFJ INCOMPETENT (EP N1→N2 present), GSV reflux. SEPARATELY, an independent tributary also shows reflux from a different source. Two independent shunt pathways. What type?",
            "response": """DECISION TREE:
Step 1: Check for EP N1→N2 (SFJ) → YES, INCOMPETENT
Step 2: Check for independent tributaries → YES, separate pathway
Step 3: Are these connected or separate? → SEPARATE, two distinct reflux sources
Step 4: Confirm both need independent ligation → YES

CLASSIFICATION: TYPE 1+2 (Combined: SFJ + Independent Tributary)
REASONING: Type 1 (SFJ incompetence) PLUS independent Type 2 pathway (tributary reflux).
LIGATION: Ligate BOTH the SFJ AND the independent tributary connection site.
Principle: Two completely separate problems, require two separate ligation sites."""
        },

        # TYPE 3 - PERFORATOR FEEDS GSV, NO SFJ
        {
            "instruction": "Classify: SFJ is COMPETENT. A perforator at Hunterian level (mid-thigh) feeds the GSV (EP N1→N2 at that location, NOT at SFJ). GSV reflux distally. No direct DV-N3 pathways. What type?",
            "response": """DECISION TREE:
Step 1: Check for SFJ incompetence (EP N1→N2 at SFJ) → NO, SFJ competent
Step 2: Check for perforator feeding GSV (EP N1→N2 elsewhere) → YES, at Hunterian
Step 3: Check for DV-N3 connections → NO, only N1→N2 present
Step 4: Identify as perforator-based → Hunterian perforator type

CLASSIFICATION: TYPE 3 (Perforator-Fed GSV)
REASONING: Perforator (not SFJ) is the incompetent source feeding GSV.
LIGATION: Ligate the perforator BOTH above and below the GSV crossing point.
Principle: CRITICAL - Must disconnect perforator completely while preserving GSV outflow via perforators above."""
        },

        # TYPE 5 - RECURRENT/RECANALIZED
        {
            "instruction": "Classify: Previous SFJ ligation evident. New recanalized pathways detected at old ligation site. Neovascularization has formed new small vessels bypassing the old ligation. What type?",
            "response": """DECISION TREE:
Step 1: Is there evidence of prior surgery? → YES, previous SFJ ligation
Step 2: Are new pathways present? → YES, recanalization
Step 3: Location of recurrence? → At the old ligation site (collateral formation)
Step 4: Mechanism? → Neovascularization, not recanalation of original vein

CLASSIFICATION: TYPE 5 (Recurrent/Recanalized Shunt)
REASONING: Post-operative recurrence with new vessel formation around old ligation.
LIGATION: Ligate recanalized pathways AND neovascular channels. Consider distal GSV ligation.
Principle: More aggressive approach needed due to collateral development."""
        },

        # ELIMINATION TEST: TYPE 3 vs TYPE 1+2
        {
            "instruction": "Differential diagnosis: Patient has reflux with GSV involvement and tributary reflux. SFJ is competent. How do you distinguish between Type 3 (perforator-fed) and Type 1+2 (independent pathways)?",
            "response": """ELIMINATION TEST FOR TYPE 3 VS TYPE 1+2:

Type 3 characteristics:
- Single N1→N2 entry point (the perforator)
- GSV reflux ORIGINATES from this perforator
- Tributaries feed FROM the GSV (secondary to GSV reflux)
- N1→N3 is NOT directly present
- Elimination: Can disconnect perforator BOTH sides and GSV would be competent above

Type 1+2 characteristics:
- Multiple independent N1→N2 entries OR N1→N3 entries
- Each pathway operates independently
- Tributaries may have separate reflux sources
- Cannot be fixed by single perforator ligation

DIAGNOSTIC KEY:
1. Identify ALL entry points (EP flows)
2. Count how many independent sources exist
3. For Type 3: Only 1 perforator source
4. For Type 1+2: Multiple independent sources (SFJ + separate tributary)

DECISION: If reflux can be traced to single perforator source → TYPE 3
If multiple independent sources → TYPE 1+2"""
        },
    ]

    return pairs


if __name__ == "__main__":
    pairs = generate_better_training_pairs()
    print(f"Generated {len(pairs)} improved training pairs")
    for i, pair in enumerate(pairs[:2], 1):
        print(f"\n--- Pair {i} ---")
        print(f"Instruction: {pair['instruction'][:100]}...")
        print(f"Response: {pair['response'][:150]}...")
