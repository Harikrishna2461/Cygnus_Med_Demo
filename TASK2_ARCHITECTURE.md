# Task 2 System Architecture Diagram

## Complete System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ULTRASOUND CLINIC WORKFLOW                         │
└──────────────────┬──────────────────────────────────────────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────┐
    │   FRONTEND: React UI (ProbeGuidance) │
    ├──────────────────────────────────────┤
    │                                      │
    │  ┌────────────────────────────────┐ │
    │  │  STEP 1: Select Sonographer    │ │
    │  │  ┌─ Dr. Sarah Chen (12yr)      │ │
    │  │  │  Specialty: Complex CHIVA   │ │
    │  │  │  ✨ 5 sessions, 124 clips   │ │
    │  │  ├─ Dr. James Okoye (4yr)      │ │
    │  │  │  Specialty: Perforators     │ │
    │  │  │  ✨ 3 sessions, 98 clips    │ │
    │  │  └─ Dr. Maria Santos (8yr)     │ │
    │  │     Specialty: Pelvic reflux   │ │
    │  │     ✨ 4 sessions, 162 clips   │ │
    │  └────────────────────────────────┘ │
    │                  │                   │
    │                  ▼                   │
    │  ┌────────────────────────────────┐ │
    │  │  STEP 2: Choose Analysis Mode  │ │
    │  │  ┌─ Single Position            │ │
    │  │  │  └─ Analyze 1 finding       │ │
    │  │  ├─ Stream Multiple            │ │
    │  │  │  └─ Real-time sequence      │ │
    │  │  └─ Analyze Previous ⭐ NEW    │ │
    │  │     └─ Review historical      │ │
    │  └────────────────────────────────┘ │
    │                  │                   │
    │                  ▼                   │
    │  ┌────────────────────────────────┐ │
    │  │  STEP 3: Receive Guidance      │ │
    │  │  "Given Dr. Sarah's preference │ │
    │  │   for longitudinal views,      │ │
    │  │   move probe medially in       │ │
    │  │   left SFJ zone (X: 0.50–0.61)│ │
    │  │   and apply Valsalva..."       │ │
    │  └────────────────────────────────┘ │
    │                                      │
    └──────────────────┬───────────────────┘
                       │
                       │ HTTP POST
                       │ /api/probe-guidance
                       │
                       ▼
    ┌──────────────────────────────────────────────────────────┐
    │         BACKEND: Flask + Python                          │
    ├──────────────────────────────────────────────────────────┤
    │                                                          │
    │  ┌────────────────────────────────────────────────────┐ │
    │  │  1. SONOGRAPHER CONTEXT BUILDER                   │ │
    │  │  Input: sonographer_id = "sono-001"              │ │
    │  │  ↓                                                │ │
    │  │  • Query sonographers table                      │ │
    │  │    → {name: "Dr. Sarah Chen",                    │ │
    │  │       experience: 12,                            │ │
    │  │       scanning_style: "top-down SFJ..."}         │ │
    │  │                                                  │ │
    │  │  • Query last 3 sessions from DB                │ │
    │  │    → [{session_id: "uuid-1",                     │ │
    │  │        total_points: 28,                         │ │
    │  │        reflux_count: 7,                          │ │
    │  │        guidance_history: [...]},                 │ │
    │  │       ...]                                       │ │
    │  │                                                  │ │
    │  │  • BUILD CONTEXT STRING:                         │ │
    │  │    "=== SONOGRAPHER PROFILE ===                 │ │
    │  │     Name: Dr. Sarah Chen                        │ │
    │  │     Experience: 12 years                        │ │
    │  │     Specialty: Complex CHIVA                    │ │
    │  │     Style: Top-down SFJ-to-knee,               │ │
    │  │            uses Valsalva routinely...           │ │
    │  │                                                  │ │
    │  │     === ANATOMICAL ZONES ===                    │ │
    │  │     RIGHT LEG:                                  │ │
    │  │       • SFJ-Knee: X=0.0931-0.475, Y=0-0.5497   │ │
    │  │       • Knee-Ankle: X=0.105-0.2947, Y=0.5497-1 │ │
    │  │       • SPJ-Ankle: X=0.2827-0.4386, Y=0.5497-1 │ │
    │  │     LEFT LEG:                                   │ │
    │  │       • SFJ-Knee: X=0.4985-0.909, Y=0-0.5497   │ │
    │  │       • Knee-Ankle: X=0.7081-0.91, Y=0.5497-1  │ │
    │  │       • SPJ-Ankle: X=0.588-0.714, Y=0.5497-1   │ │
    │  │                                                  │ │
    │  │     === SESSION HISTORY ===                     │ │
    │  │     Session 1 (2025-12-02): 28 clips, 7 reflux │ │
    │  │       [RP] Move probe medially to SFJ...        │ │
    │  │       [RP] Scan tributary at mid-calf...        │ │
    │  │     Session 2 (2025-11-15): 35 clips, 12 reflux│ │
    │  │       [RP] Locate pelvic origin reflux...       │ │
    │  │     ..."                                         │ │
    │  │                                                  │ │
    │  │  OUTPUT: Rich personalization context            │ │
    │  └────────────────────────────────────────────────┘ │
    │                       │                             │
    │                       ▼                             │
    │  ┌────────────────────────────────────────────────┐ │
    │  │  2. MEDICAL CONTEXT RETRIEVAL (RAG)            │ │
    │  │  • Query Qdrant for:                           │ │
    │  │    "SFJ reflux guidance for Type 1 pattern"    │ │
    │  │  • Retrieve top 3 relevant chunks:             │ │
    │  │    - Clinical guidelines for SFJ ligation      │ │
    │  │    - Probe positioning at saphenofemoral       │ │
    │  │    - Valsalva maneuver technique               │ │
    │  │  ✓ Context enriched with clinical knowledge    │ │
    │  └────────────────────────────────────────────────┘ │
    │                       │                             │
    │                       ▼                             │
    │  ┌────────────────────────────────────────────────┐ │
    │  │  3. LLM PROMPT CONSTRUCTION                    │ │
    │  │  Combine:                                      │ │
    │  │  ┌─ Clinical data (flow type, duration, pos) │ │
    │  │  ├─ Anatomical zone description              │ │
    │  │  ├─ Medical knowledge (RAG chunks)           │ │
    │  │  ├─ SONOGRAPHER PERSONALIZATION              │ │
    │  │  │  "This sonographer prefers...",           │ │
    │  │  │   "Their style is...",                    │ │
    │  │  │   "Past sessions show..."                 │ │
    │  │  └─ Specific zone coordinates                │ │
    │  │                                               │ │
    │  │  Final Prompt Fragment:                       │ │
    │  │  "ULTRASOUND GUIDANCE                         │ │
    │  │   REFLUX at SFJ-Knee (left leg, N1→N2)       │ │
    │  │   Position: X=0.65, Y=0.08                   │ │
    │  │   → upper-to-mid left thigh                  │ │
    │  │                                               │ │
    │  │   [SONOGRAPHER CONTEXT INSERTED HERE]        │ │
    │  │   [MEDICAL KNOWLEDGE INSERTED HERE]          │ │
    │  │                                               │ │
    │  │   TASK: Generate ONE personalized line        │ │
    │  │   Adapt to sonographer's style & zone coords" │ │
    │  └────────────────────────────────────────────────┘ │
    │                       │                             │
    │                       ▼                             │
    │  ┌────────────────────────────────────────────────┐ │
    │  │  4. LLM INFERENCE (Groq)                       │ │
    │  │  Model: llama-3.1-70b-versatile                │ │
    │  │  Temperature: 0.2 (deterministic)              │ │
    │  │  Max tokens: 80 (concise)                      │ │
    │  │                                                │ │
    │  │  ▶ Process:                                    │ │
    │  │  1. Analyze sonographer profile               │ │
    │  │  2. Reference their scanning style            │ │
    │  │  3. Note probe position zone                  │ │
    │  │  4. Consider past session patterns            │ │
    │  │  5. Generate adapted recommendation           │ │
    │  │                                                │ │
    │  │  Output: "Given Dr. Sarah's preference for    │ │
    │  │           longitudinal views, move probe      │ │
    │  │           medially in left SFJ zone           │ │
    │  │           (X: 0.50–0.61, Y: 0.05–0.15)        │ │
    │  │           and apply Valsalva to confirm       │ │
    │  │           reflux duration as you typically    │ │
    │  │           do. This follows your pattern       │ │
    │  │           from past sessions."                │ │
    │  │                                                │ │
    │  │  ✨ FULLY PERSONALIZED GUIDANCE                │ │
    │  └────────────────────────────────────────────────┘ │
    │                       │                             │
    │                       ▼                             │
    │  ┌────────────────────────────────────────────────┐ │
    │  │  5. RESPONSE & SESSION PERSISTENCE             │ │
    │  │  Return to frontend:                           │ │
    │  │  {                                             │ │
    │  │    "guidance_instruction": "Given Dr. Sarah...", │
    │  │    "flow_type": "RP",                          │ │
    │  │    "anatomical_location": "Left SFJ zone",     │ │
    │  │    "reflux_duration": "1.1s",                  │ │
    │  │    "confidence": 0.88,                         │ │
    │  │    "probe_zone": "SFJ - left thigh"            │ │
    │  │  }                                             │ │
    │  │                                                │ │
    │  │  Auto-save to database:                        │ │
    │  │  • sonographer_id                             │ │
    │  │  • session_date                               │ │
    │  │  • guidance_history (append)                  │ │
    │  │  • flow_type, instruction                     │ │
    │  │  ✓ Feeds into future personalization!         │ │
    │  └────────────────────────────────────────────────┘ │
    └──────────────────────┬───────────────────────────────┘
                           │
                           │ JSON Response
                           │
                           ▼
    ┌──────────────────────────────────────────────────────┐
    │        DATABASE: SQLite (mlops_metrics.db)           │
    ├──────────────────────────────────────────────────────┤
    │                                                      │
    │  sonographers TABLE:                                │
    │  ┌──────────────────────────────────────────────┐  │
    │  │ ID       │ Name        │ Experience │ Style  │  │
    │  │──────────┼─────────────┼────────────┼────────│  │
    │  │ sono-001 │ Sarah Chen  │ 12 years   │ TDN... │  │
    │  │ sono-002 │ James Okoye │ 4 years    │ BtU... │  │
    │  │ sono-003 │ Maria Santos│ 8 years    │ BLT... │  │
    │  └──────────────────────────────────────────────┘  │
    │                                                      │
    │  sonographer_sessions TABLE:                        │
    │  ┌──────────────────────────────────────────────┐  │
    │  │ session_id │ sono_id │ date  │ clips │ reflux  │  │
    │  │────────────┼─────────┼───────┼───────┼────────│  │
    │  │ uuid-1     │ sono-001│ Dec 2 │ 28    │ 7      │  │
    │  │ uuid-2     │ sono-001│ Nov 15│ 35    │ 12     │  │
    │  │ uuid-3     │ sono-002│ Dec 1 │ 32    │ 10     │  │
    │  └──────────────────────────────────────────────┘  │
    │                                                      │
    │  guidance_history (STORED AS JSON):                 │
    │  ┌──────────────────────────────────────────────┐  │
    │  │ {                                            │  │
    │  │   "point": 1,                                │  │
    │  │   "flow_type": "RP",                         │  │
    │  │   "instruction": "Move probe medially...",   │  │
    │  │   "clinical_reason": "Reflux detected..."    │  │
    │  │ }                                            │  │
    │  └──────────────────────────────────────────────┘  │
    │                                                      │
    │  ✓ Persistent storage                              │
    │  ✓ Historical data accumulates                      │
    │  ✓ Feeds into future guidance                       │
    └──────────────────────────────────────────────────────┘
```

---

## Data Flow: Complete Example

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ EXAMPLE: Dr. Sarah Chen analyzes a reflux finding                          │
└─────────────────────────────────────────────────────────────────────────────┘

STEP 1: USER INPUT
───────────────────
Sonographer: Dr. Sarah Chen (sono-001)
Ultrasound Data:
  {
    "flow": "RP",
    "step": "SFJ-Knee",
    "reflux_duration": 1.1,
    "posXRatio": 0.65,
    "posYRatio": 0.08,
    "legSide": "left",
    "fromType": "N1",
    "toType": "N2",
    "confidence": 0.88
  }


STEP 2: DATABASE QUERY
──────────────────────
Query sonographers WHERE id = 'sono-001'
  ↓
Retrieve:
  Name: "Dr. Sarah Chen"
  Title: "Senior Vascular Sonographer"
  Experience: 12 years
  Specialty: "Complex CHIVA & bilateral lower limb assessment"
  Scanning Style: "Starts assessment at the SFJ (groin) and traces the GSV 
                   distally to the knee. Uses Valsalva manoeuvre routinely 
                   for valve competence testing. Prefers longitudinal views 
                   at SFJ before switching to transverse for diameter measurement..."

Query sonographer_sessions WHERE sonographer_id = 'sono-001' LIMIT 3
  ↓
Retrieve Last 3 Sessions:
  Session 1 (Dec 2):  28 clips, 7 reflux, "Type 1 + Type 3", guidance_history = [...]
  Session 2 (Nov 15): 35 clips, 12 reflux, "Complex perforator", guidance_history = [...]
  Session 3 (Oct 28): 31 clips, 6 reflux, "Bilateral Type 1", guidance_history = [...]


STEP 3: CONTEXT BUILDING
─────────────────────────
Construct context string:
```
=== SONOGRAPHER PROFILE ===
Name: Dr. Sarah Chen (Senior Vascular Sonographer)
Experience: 12 years | Specialty: Complex CHIVA & bilateral lower limb assessment
Habitual scanning style: Starts assessment at the SFJ (groin) and traces the 
GSV distally to the knee. Uses Valsalva manoeuvre routinely for valve 
competence testing. Prefers longitudinal views at SFJ before switching to 
transverse for diameter measurement. Tends to revisit reflux points twice 
to confirm duration. Typically scans the right leg first, then mirrors 
the pattern on the left.

=== ANATOMICAL SCANNING ZONES ===
RIGHT LEG (left side of image):
  • SFJ-Knee Zone: X=0.0931-0.475, Y=0-0.5497
  • Knee-Ankle Zone: X=0.105-0.2947, Y=0.5497-1
  • SPJ-Ankle Zone: X=0.2827-0.4386, Y=0.5497-1
LEFT LEG (right side of image):
  • SFJ-Knee Zone: X=0.4985-0.909, Y=0-0.5497
  • Knee-Ankle Zone: X=0.7081-0.91, Y=0.5497-1
  • SPJ-Ankle Zone: X=0.588-0.714, Y=0.5497-1

=== PREVIOUS SESSION HISTORY ===
Session 1 (2025-12-02): 28 clips, 7 reflux detections
  Summary: Type 1 + Type 3 reflux patterns
  [RP] Move probe medially to locate SFJ junction
  [RP] Scan tributary at mid-calf level

Session 2 (2025-11-15): 35 clips, 12 reflux detections
  Summary: Complex perforator loop detected
  [RP] Locate pelvic origin reflux above SFJ
  [RP] Trace perforator cluster in calf zone
```


STEP 4: ZONE DETECTION
──────────────────────
Function: detect_zone(leg='left', px=0.65, py=0.08)
Result: "SFJ (groin) — saphenofemoral junction level"

Reason: py=0.08 ≤ 0.098 → groin level


STEP 5: RAG RETRIEVAL
─────────────────────
Query: "Ultrasound probe guidance for RP at SFJ-Knee location with N1 to N2 
        network. SFJ groin-level zone. Duration: 1.1s. Confidence: 0.88"

Qdrant Search Results (top chunks):
  Chunk 1: "Saphenofemoral junction ligation requires accurate probe positioning 
           at the groin level with longitudinal transducer approach to identify 
           the junction anatomy and confirm reflux duration..."
  Chunk 2: "Valsalva maneuver technique: instruct patient to bear down while 
           scanning - reflux >0.5s is significant for treatment planning..."
  Chunk 3: "Type 1 reflux classification: N1→N2 flow indicates saphenofemoral 
           incompetence; ligation at SFJ junction is primary treatment..."


STEP 6: LLM PROMPT ASSEMBLY
────────────────────────────
```
ULTRASOUND GUIDANCE
REFLUX at SFJ-Knee (left leg, N1→N2)
Probe position: posXRatio=0.65, posYRatio=0.08 → SFJ (groin) level
(System coordinates: (0,0)=top-left, (1,1)=bottom-right)

=== MEDICAL KNOWLEDGE BASE (retrieved via RAG) ===
[RAG chunks inserted here...]

=== SONOGRAPHER PROFILE ===
[Full context from STEP 3 inserted here]

EXAMPLES of CORRECT 1-line guidance:
- "Move probe medially to locate SFJ junction"
- "Position longitudinally at groin level for diameter measurement"
- "Apply Valsalva to confirm reflux duration"

TASK: Generate ONE personalised line. Action verb + anatomical target. 
Use exact anatomical zone and coordinates above.
IMPORTANT: Adapt guidance to this sonographer's experience level and known 
scanning style. Reference how they typically approach similar findings.
Confirmed location: saphenofemoral junction (SFJ) at groin

<guidance_instruction>Write one clear instruction</guidance_instruction>
```


STEP 7: LLM INFERENCE (Groq)
─────────────────────────────
Processing:
  1. Analyze sonographer name → "Dr. Sarah Chen"
  2. Recognize profile → Senior, 12 years, systematic, top-down
  3. Note probe position → X=0.65, Y=0.08 = SFJ groin level
  4. Review past sessions → Always starts at SFJ, uses Valsalva, prefers longitudinal
  5. Consider zone coordinates → Left SFJ zone: X 0.4985-0.909, Y 0-0.5497
  6. Generate personalized response...

LLM Output:
───────────
"Given your preference for longitudinal views at the SFJ and your systematic 
top-down approach, position the probe medially in the left SFJ zone 
(X: 0.50–0.61, Y: 0.05–0.15) using longitudinal transducer orientation, 
then apply Valsalva maneuver to confirm reflux duration as you typically do."


STEP 8: RESPONSE TO FRONTEND
─────────────────────────────
JSON Response:
{
  "guidance_instruction": "Given your preference for longitudinal views at 
                          the SFJ and your systematic top-down approach, 
                          position the probe medially in the left SFJ zone 
                          (X: 0.50–0.61, Y: 0.05–0.15) using longitudinal 
                          transducer orientation, then apply Valsalva 
                          maneuver to confirm reflux duration as you 
                          typically do.",
  "clinical_reason": "N1→N2 reflux at SFJ indicates saphenofemoral 
                     incompetence. Valsalva testing confirms duration for 
                     treatment planning.",
  "flow_type": "RP",
  "anatomical_location": "Left SFJ zone (groin level)",
  "reflux_duration": "1.1s",
  "confidence": 0.88,
  "probe_zone": "SFJ - left groin",
  "analysis_time_ms": 342
}


STEP 9: FRONTEND DISPLAY
────────────────────────
User sees:
```
════════════════════════════════════════════════════════════════
    🔍 Clinical Assessment
════════════════════════════════════════════════════════════════
Flow Type:                    Anatomical Location:
❌ Reflux (RP)               Left SFJ zone (groin level)

Reflux Duration:             Confidence Level:
1.1 seconds                  88%

════════════════════════════════════════════════════════════════
    📍 Probe Guidance Instruction
════════════════════════════════════════════════════════════════

Given your preference for longitudinal views at the SFJ and your 
systematic top-down approach, position the probe medially in the left 
SFJ zone (X: 0.50–0.61, Y: 0.05–0.15) using longitudinal transducer 
orientation, then apply Valsalva maneuver to confirm reflux duration 
as you typically do.

════════════════════════════════════════════════════════════════
    📋 Clinical Reasoning
════════════════════════════════════════════════════════════════

N1→N2 reflux at SFJ indicates saphenofemoral incompetence. Valsalva 
testing confirms duration for treatment planning.

════════════════════════════════════════════════════════════════
```


STEP 10: DATABASE PERSISTENCE
──────────────────────────────
Auto-save session update:
  sonographer_id: "sono-001"
  session_date: 2026-04-15 14:30:45
  mode: "single"
  total_points: 1 (incremented)
  reflux_count: 1 (incremented)
  guidance_history append:
    {
      "point": 1,
      "flow_type": "RP",
      "instruction": "Given your preference for...",
      "clinical_reason": "N1→N2 reflux..."
    }

✓ NEXT TIME Dr. Sarah analyzes something:
  → System retrieves these 3 sessions + new data
  → Personalization improves further
  → Digital twin gets more accurate
```

---

## Session Analysis Flow (Analyze Previous Session Mode)

```
┌────────────────────────────────────────────────────────────┐
│ USER: Clicks "Analyze Previous Session" mode               │
└────────────────────┬───────────────────────────────────────┘
                     ▼
    ┌────────────────────────────────────────────┐
    │ UI: Display last 5 sessions as cards       │
    │ ┌─────────────────────────────────────┐   │
    │ │ 2 Dec 2025 | 14:30                 │   │
    │ │ 28 clips · 7 reflux                │   │
    │ │ "Type 1 + Type 3 patterns found"   │   │
    │ └─────────────────────────────────────┘   │
    │ ┌─────────────────────────────────────┐   │
    │ │ 15 Nov 2025 | 11:15                │ ◄─┤─ USER SELECTS THIS
    │ │ 35 clips · 12 reflux               │   │
    │ │ "Complex perforator loop"          │   │
    │ └─────────────────────────────────────┘   │
    │ ┌─────────────────────────────────────┐   │
    │ │ 28 Oct 2025 | 09:45                │   │
    │ │ 31 clips · 6 reflux                │   │
    │ │ "Bilateral Type 1 assessment"      │   │
    │ └─────────────────────────────────────┘   │
    │ ... (2 more sessions)                     │
    │                                            │
    │ [🔍 Analyze Selected Session]             │
    └────────────────────────────────────────────┘
                     ▼
    ┌────────────────────────────────────────────┐
    │ FRONTEND: handleAnalyzePreviousSession()   │
    │ ┌────────────────────────────────────────┐ │
    │ │ 1. Extract selected session data        │ │
    │ │    session_id: "uuid-2"                │ │
    │ │    session_date: "2025-11-15T11:15"   │ │
    │ │    total_points: 35                   │ │
    │ │    reflux_count: 12                   │ │
    │ │    guidance_history: [35 items]       │ │
    │ └────────────────────────────────────────┘ │
    │                                            │
    │ ┌────────────────────────────────────────┐ │
    │ │ 2. Analyze statistics                   │ │
    │ │    Total points: 35                    │ │
    │ │    Reflux (RP): 12 → 34.3%            │ │
    │ │    Normal (EP): 23 → 65.7%            │ │
    │ │    Duration: 35 × 0.5s = ~17.5 min   │ │
    │ └────────────────────────────────────────┘ │
    │                                            │
    │ ┌────────────────────────────────────────┐ │
    │ │ 3. Extract key findings                │ │
    │ │    Top reflux locations (sample):      │ │
    │ │    • "Locate pelvic origin reflux..."  │ │
    │ │    • "Trace perforator cluster..."     │ │
    │ │    • "Position behind knee for..."     │ │
    │ │                                        │ │
    │ │    Top normal areas (sample):          │ │
    │ │    • "Continue scanning distally..."   │ │
    │ │    • "Assess tributary junction..."    │ │
    │ │    • "Move to ankle perforators..."    │ │
    │ └────────────────────────────────────────┘ │
    │                                            │
    │ ┌────────────────────────────────────────┐ │
    │ │ 4. Build sonographer insights          │ │
    │ │    Based on profile (Dr. Sarah):       │ │
    │ │    • "Pattern: top-down systematic"    │ │
    │ │    • "Strength: thorough bilateral"    │ │
    │ │    • "Habit: revisits doubly"          │ │
    │ │    • "Next: use patterns for future"   │ │
    │ └────────────────────────────────────────┘ │
    │                                            │
    │ ┌────────────────────────────────────────┐ │
    │ │ 5. Set sessionComparison state + UI    │ │
    │ │    ✓ Ready to display                  │ │
    │ └────────────────────────────────────────┘ │
    └────────────────────┬───────────────────────┘
                         ▼
    ┌────────────────────────────────────────────┐
    │ UI: Display Session Analysis Report         │
    │                                            │
    │ ┌──────────────┬───────────┬──────────┐   │
    │ │ 35 Clips     │ 12 Reflux │ 23 Normal│   │
    │ │ Analyzed     │ Found     │ Areas    │   │
    │ └──────────────┴───────────┴──────────┘   │
    │                                            │
    │ 📊 Reflux: 34.3%  │  Normal: 65.7%      │
    │                                            │
    │ 🔎 Key Findings:                          │
    │    Reflux Locations        Normal Areas   │
    │    • Pelvic origin         • GSV trunk   │
    │    • Perforator cluster    • Ankle zone  │
    │    • Behind knee           • SPJ check   │
    │                                            │
    │ 💡 Personalized Insights:                 │
    │    Based on Dr. Sarah's style:           │
    │    • Pattern: Top-down SFJ-to-knee       │
    │    • Strength: Thorough bilateral        │
    │    • Habit: Revisits points twice        │
    │    • Recommendation: Use for future      │
    │                                            │
    │ 📈 Complete Guidance History:             │
    │    [Scrollable list of 35 items]         │
    │    Position 1:  [RP] "Locate pelvic..." │
    │    Position 2:  [RP] "Trace perpof..." │
    │    ...                                    │
    │    Position 35: [EP] "Ankle assessment" │
    │                                            │
    └────────────────────────────────────────────┘
```

---

## Personalization Degrees

```
┌─────────────────────────────────────────────────────────────┐
│         PERSONALIZATION GRADIENT                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  NO PERSONALIZATION (Generic):                             │
│  "Move probe to the vein location"                         │
│                                                             │
│       ↓  ADD SONOGRAPHER NAME                              │
│                                                             │
│  "Dr. Sarah, move probe to the vein"                       │
│                                                             │
│       ↓  ADD SCANNING STYLE                                │
│                                                             │
│  "Given your top-down approach, start at SFJ and work     │
│   distally"                                                │
│                                                             │
│       ↓  ADD ZONE COORDINATES                              │
│                                                             │
│  "In left SFJ zone (X: 0.50–0.61, Y: 0.05–0.15),         │
│   move probe medially"                                     │
│                                                             │
│       ↓  ADD SESSION HISTORY                               │
│                                                             │
│  "You've detected reflux in calf zone in past sessions,    │
│   so focus probe medially in left SFJ zone                │
│   (X: 0.50–0.61, Y: 0.05–0.15)"                          │
│                                                             │
│       ↓  ADD PREFERRED TECHNIQUES                          │
│                                                             │
│  "Given your preference for longitudinal views and         │
│   Valsalva technique, position probe medially in left     │
│   SFJ zone (X: 0.50–0.61, Y: 0.05–0.15), apply          │
│   longitudinal orientation, then Valsalva as you          │
│   typically do"                                            │
│                                                             │
│       ↓  ADD CONTEXTUAL INTELLIGENCE                       │
│                                                             │
│  "Given your preference for longitudinal views at the SFJ │
│   and your systematic top-down approach, position the     │
│   probe medially in the left SFJ zone (X: 0.50–0.61,     │
│   Y: 0.05–0.15) using longitudinal transducer             │
│   orientation, then apply Valsalva maneuver to confirm    │
│   reflux duration as you typically do. This follows your  │
│   proven pattern from past sessions where you             │
│   successfully detected Type 1 + Type 3 patterns."        │
│                                                             │
│  = FULLY PERSONALIZED GUIDANCE ✨                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Takeaways

✅ **Three-Layer Architecture**:
1. Frontend (React UI) → React UI with 3 analysis modes
2. Backend (Flask) → Orchestrates personalization, medical knowledge, LLM
3. Database (SQLite) → Stores profiles, sessions, history

✅ **Personalization Pipeline**:
1. Select sonographer → Retrieve profile + history
2. Anatomical zone detection → Map coordinates to zones
3. Medical knowledge retrieval → RAG from Qdrant
4. Context building → Combine all data
5. LLM generation → Personalized guidance
6. Session persistence → Feeds future guidance

✅ **Three Analysis Modes**:
1. Single Position → Quick guidance on one finding
2. Stream Multiple → Real-time multi-position analysis
3. Analyze Previous Session → Historical pattern review

✅ **Digital Twin Effect**:
- More sessions → Richer context
- Richer context → Better personalization
- Better personalization → More useful guidance
- Cycle repeats infinitely

