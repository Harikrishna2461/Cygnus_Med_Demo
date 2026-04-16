# Task 2: LLM-Based Personalized Sonographer Guidance System

## Overview

Task 2 implements a **digital twin sonographer profiling system** that captures individual sonographer scanning patterns and provides personalized probe guidance. Combined with standardized positional information for anatomical zones, this creates an intelligent, adaptive guidance system.

---

## Part 1: Positional Information System

### Anatomical Coordinate System

All ultrasound probe positions are normalized to a standard coordinate system:
- **Origin (0,0)**: Top-left corner of the image
- **Extent (1,1)**: Bottom-right corner of the image
- **Width**: Normalized X-coordinates from 0 to 1
- **Height**: Normalized Y-coordinates from 0 to 1

### Anatomical Zones by Leg

#### **RIGHT LEG** (appears on left side of ultrasound image)

| Zone | X Range | Y Range | Anatomical Region |
|------|---------|---------|-------------------|
| **SFJ-Knee** | 0.0931–0.475 | 0–0.5497 | Saphenofemoral junction to knee (thigh) |
| **Knee-Ankle** | 0.105–0.2947 | 0.5497–1.0 | Below-knee/mid-to-distal calf |
| **SPJ-Ankle** | 0.2827–0.4386 | 0.5497–1.0 | Saphenopopliteal junction to ankle (posterior calf) |

#### **LEFT LEG** (appears on right side of ultrasound image)

| Zone | X Range | Y Range | Anatomical Region |
|------|---------|---------|-------------------|
| **SFJ-Knee** | 0.4985–0.909 | 0–0.5497 | Saphenofemoral junction to knee (thigh) |
| **Knee-Ankle** | 0.7081–0.91 | 0.5497–1.0 | Below-knee/mid-to-distal calf |
| **SPJ-Ankle** | 0.588–0.714 | 0.5497–1.0 | Saphenopopliteal junction to ankle (posterior calf) |

### How LLM Uses Positional Information

When the LLM generates probe guidance:
1. **Input**: `posXRatio`, `posYRatio` from ultrasound data
2. **Processing**: Maps coordinates to anatomical zone description
3. **Output**: Specific, zone-aware guidance like "Move probe medially in the left SFJ zone (X: 0.50–0.61 range)"

**Example Zone Detection:**
- Position: left leg, X=0.65, Y=0.22
- Detected Zone: "upper-to-mid left thigh (Hunterian canal region), GSV medial course"
- Guidance: Adapts to this specific anatomical location

---

## Part 2: Digital Twin Sonographer Profiling

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TASK-2 UI (React)                        │
├─────────────────────────────────────────────────────────────┤
│  1. Sonographer Profiles    3. Session Selection             │
│  2. Session List            4. Analysis Mode                 │
└──────────────┬──────────────────────────────────┬────────────┘
               │                                  │
        [Single Position]              [Stream Multiple]
        [Analyze Previous]                 Analysis
               │                                  │
               └──────────────┬──────────────────┘
                              │
        ┌─────────────────────▼──────────────────────┐
        │   Backend API (/api/probe-guidance)        │
        ├────────────────────────────────────────────┤
        │  • Retrieve sonographer profile            │
        │  • Load historical session data (last 3)   │
        │  • Build personalized context              │
        │  • Include positional zone information     │
        └─────────────────────┬──────────────────────┘
                              │
        ┌─────────────────────▼──────────────────────┐
        │         LLM Prompt with Context:           │
        │  • Sonographer name & specialty            │
        │  • Experience level                        │
        │  • Known scanning style & habits           │
        │  • Past reflux detection patterns          │
        │  • Anatomical zone coordinates             │
        │  • Medical knowledge (RAG retrieval)       │
        └─────────────────────┬──────────────────────┘
                              │
        ┌─────────────────────▼──────────────────────┐
        │      LLM Output (Personalized Guidance)    │
        │  "Given Dr. Sarah's preference for        │
        │   longitudinal views, move probe...        │
        │   medially in the left SFJ zone"           │
        └────────────────────────────────────────────┘
```

### Three Pre-Seeded Sonographer Profiles

#### **1. Dr. Sarah Chen**
- **ID**: sono-001
- **Specialty**: Complex CHIVA & bilateral lower limb assessment
- **Experience**: 12 years
- **Scanning Style**: Top-down SFJ-to-knee mapping; uses Valsalva routinely; prefers longitudinal views at SFJ; revisits reflux points twice to confirm duration
- **Tendency**: Highly systematic; starts right leg first

#### **2. Dr. James Okoye**
- **ID**: sono-002
- **Specialty**: Type 2 perforator mapping & calf vein assessment
- **Experience**: 4 years
- **Scanning Style**: Bottom-up ankle-to-proximal approach; thorough calf perforator mapping; occasional misses of Hunterian perforators; frequent compression maneuver use
- **Tendency**: Detail-oriented but sometimes less comprehensive at high SFJ level

#### **3. Dr. Maria Santos**
- **ID**: sono-003
- **Specialty**: Pelvic origin reflux, SSV & SPJ assessment
- **Experience**: 8 years
- **Scanning Style**: Bilateral simultaneous comparison; posterior-first approach at SPJ; expert at complex Type 4/5 patterns; applies elimination test routinely; careful SSV documentation
- **Tendency**: Comprehensive bilateral assessment before conclusions

### Workflow: Session Analysis Example

#### **Step 1: Sonographer Selects Profile**
User navigates to **Task-2 → Probe Guidance** and selects a sonographer name from the three available profiles.

```
UI Display:
┌──────────────────────────────────────────┐
│        🎯 Probe Guidance                 │
│     Select Sonographer                   │
├──────────────────────────────────────────┤
│  ┌──────────────────────────────────────┐│
│  │  👤 Dr. Sarah Chen                   ││
│  │  Senior Vascular Sonographer         ││
│  │  12 years | Complex CHIVA            ││
│  │  ✨ 5 Sessions | Last: 2 Dec 2025    ││
│  │     ▶ Start Session →                ││
│  └──────────────────────────────────────┘│
│  ┌──────────────────────────────────────┐│
│  │  👤 Dr. James Okoye                  ││
│  │  Vascular Sonographer                ││
│  │  4 years | Perforator Mapping        ││
│  │  ✨ 3 Sessions | Last: 28 Nov 2025   ││
│  │     ▶ Start Session →                ││
│  └──────────────────────────────────────┘│
└──────────────────────────────────────────┘
```

#### **Step 2: Enter Sonographer Session**
After clicking a profile, user sees:
- **Past Sessions Tab**: 3 most recent sessions with summaries
- **Analysis Modes**:
  - Single Position: Get guidance for one ultrasound finding
  - Stream Multiple: Process sequence of findings with real-time guidance
  - **Analyze Previous Session**: Review historical session data

#### **Step 3: Analyze Previous Session (New Feature)**
User clicks **"📊 Analyze Previous Session"** mode:

```
UI Display:
┌──────────────────────────────────────────────────────────┐
│  📊 Analyze Previous Sonographer Session                 │
│                                                          │
│  Select Session to Analyze:                             │
│  ┌─────────────────────────────────────────────────────┐│
│  │ 2 December 2025                                     ││
│  │ 28 clips · 7 reflux detections                      ││
│  │ "Bilateral GSV reflux from SFJ; Type 1 + Type 3..." ││
│  └─────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────┐│
│  │ 15 November 2025                                    ││
│  │ 35 clips · 12 reflux detections                     ││
│  │ "Complex perforator loop; P→N2→N3 pathway..."       ││
│  └─────────────────────────────────────────────────────┘│
│                                                          │
│            [🔍 Analyze Selected Session]               │
└──────────────────────────────────────────────────────────┘
```

#### **Step 4: View Session Analysis**
After selection, user sees comprehensive breakdown:

```
UI Display:
┌──────────────────────────────────────────────────────────┐
│  📋 Session Overview                                     │
│  ┌──────────────────────────────────────────────────────┐│
│  │ 28 Clips Analyzed  │  7 Reflux Found  │  21 Normal  ││
│  │    (3.2 min)       │     (25%)        │    (75%)    ││
│  └──────────────────────────────────────────────────────┘│
│                                                          │
│  🔎 Key Findings:                                       │
│    Reflux Locations:              Normal Areas:         │
│    • Move probe medially at SFJ   • Left leg GSV trunk  │
│    • Scan tributary at mid-calf   • Right ankle region  │
│    • Check posterior knee zone    • SPJ bilateral check │
│                                                          │
│  💡 Personalized Insights:                              │
│    Based on Dr. Sarah's scanning style and this session ││
│    • Pattern: Prefers longitudinal SFJ assessment       ││
│    • Strength: Thorough bilateral comparison            ││
│    • Habit: Revisits findings twice for confirmation    ││
│    • Next: Use patterns for personalized guidance       ││
│                                                          │
│  📈 Complete Guidance History: (28 items shown)         │
│    ✓ Continue scanning GSV distally to knee            ││
│    ❌ Move probe medially to locate SFJ junction       ││
│    ✓ Assess next tributary junction below level        ││
│    ... (25 more items)                                 ││
└──────────────────────────────────────────────────────────┘
```

### Database Schema

#### **sonographers**
```sql
CREATE TABLE sonographers (
    id TEXT PRIMARY KEY,
    name TEXT,
    title TEXT,
    specialty TEXT,
    experience_years INTEGER,
    avatar_color TEXT,
    scanning_style TEXT,
    created_at TIMESTAMP
);
```

#### **sonographer_sessions**
```sql
CREATE TABLE sonographer_sessions (
    session_id TEXT PRIMARY KEY,
    sonographer_id TEXT,
    session_date TIMESTAMP,
    mode TEXT,
    total_points INTEGER,
    reflux_count INTEGER,
    guidance_history TEXT (JSON),
    session_summary TEXT,
    created_at TIMESTAMP
);
```

### LLM Personalization Pipeline

**Input Data Passed to LLM:**

```python
sonographer_context = """
=== SONOGRAPHER PROFILE (personalise guidance to this individual) ===
Name: Dr. Sarah Chen (Senior Vascular Sonographer)
Experience: 12 years | Specialty: Complex CHIVA & bilateral lower limb assessment
Habitual scanning style: Starts assessment at the SFJ (groin) and traces the GSV 
distally to the knee. Uses Valsalva manoeuvre routinely for valve competence testing. 
Prefers longitudinal views at SFJ before switching to transverse for diameter measurement. 
Tends to revisit reflux points twice to confirm duration. Typically scans the right leg 
first, then mirrors the pattern on the left.

=== ANATOMICAL SCANNING ZONES (normalized coordinates) ===
RIGHT LEG (left side of image):
  • SFJ-Knee Zone: X=0.0931-0.475, Y=0-0.5497
  • Knee-Ankle Zone: X=0.105-0.2947, Y=0.5497-1
  • SPJ-Ankle Zone: X=0.2827-0.4386, Y=0.5497-1

LEFT LEG (right side of image):
  • SFJ-Knee Zone: X=0.4985-0.909, Y=0-0.5497
  • Knee-Ankle Zone: X=0.7081-0.91, Y=0.5497-1
  • SPJ-Ankle Zone: X=0.588-0.714, Y=0.5497-1

=== PREVIOUS SESSION HISTORY (last 3) ===
Session 1 (2 Dec 2025): 28 clips, 7 reflux detections, mode=stream
  Summary: Bilateral GSV reflux from SFJ; Type 1 + Type 3 patterns
  [RP] Move probe medially to locate SFJ junction
  [RP] Scan tributary at mid-calf level
  
Session 2 (15 Nov 2025): 35 clips, 12 reflux detections, mode=stream
  Summary: Complex perforator loop; P→N2→N3 pathway detected
  [RP] Locate pelvic origin reflux above SFJ
  [RP] Trace perforator cluster in calf zone

USE THIS CONTEXT: Tailor probe guidance to match this sonographer's experience level, 
known style, and past scanning patterns. Reference their habits where relevant.
When providing probe position guidance, use the normalized coordinate zones above to be specific.
"""
```

**LLM Generates:**

```
Given Dr. Sarah's preference for longitudinal views at the SFJ and her systematic 
top-down approach, scan the left leg SFJ-Knee zone (X: 0.50–0.61, Y: 0.05–0.15) 
using longitudinal transducer orientation, then apply Valsalva maneuver to confirm 
valve competence as she typically does.
```

### Frontend Components Modified

#### **SonographerProfiles.js**
- Displays all 3 seeded sonographer profiles
- Shows session count and last session date
- Routes to `/probe/:sonographerId` on click

#### **ProbeGuidance.js** (Enhanced)
- **Three Analysis Modes**:
  1. **Single Position**: Traditional single ultrasound finding analysis
  2. **Stream Multiple**: Sequential processing with real-time guidance
  3. **Analyze Previous Session**: NEW - Review and compare historical sessions
  
- **New UI Components**:
  - Session selection cards with filters
  - Session statistics (clinic clips, reflux %, normal %)
  - Personalized insights derived from scanning pattern
  - Complete guidance history from past sessions

### Backend Endpoints

#### **GET /api/sonographers**
Returns all 3 seeded sonographer profiles with session counts

**Response:**
```json
[
  {
    "id": "sono-001",
    "name": "Dr. Sarah Chen",
    "title": "Senior Vascular Sonographer",
    "specialty": "Complex CHIVA & bilateral lower limb assessment",
    "experience_years": 12,
    "avatar_color": "#3b82f6",
    "session_count": 5,
    "last_session_date": "2025-12-02T14:30:00"
  },
  // ... more sonographers
]
```

#### **GET /api/sonographers/{sonographer_id}/sessions?limit=5**
Returns last N sessions for sonographer with full guidance history

**Response:**
```json
[
  {
    "session_id": "uuid-123",
    "sonographer_id": "sono-001",
    "session_date": "2025-12-02T14:30:00",
    "mode": "stream",
    "total_points": 28,
    "reflux_count": 7,
    "session_summary": "Bilateral GSV reflux from SFJ; Type 1 + Type 3 patterns",
    "guidance_history": [
      {
        "point": 1,
        "flow_type": "RP",
        "instruction": "Move probe medially to locate SFJ junction",
        "clinical_reason": "Reflux detected in saphenofemoral zone"
      },
      // ... more guidance items
    ]
  },
  // ... more sessions
]
```

#### **POST /api/probe-guidance** (Enhanced)
Now includes sonographer context in LLM prompt

**Request:**
```json
{
  "ultrasound_data": {
    "flow": "RP",
    "step": "SFJ-Knee",
    "reflux_duration": 1.1,
    "posXRatio": 0.65,
    "posYRatio": 0.08,
    "legSide": "left",
    "fromType": "N1",
    "toType": "N2",
    "confidence": 0.88
  },
  "sonographer_id": "sono-001"
}
```

**Response:**
```json
{
  "guidance_instruction": "Given your preference for longitudinal SFJ views, 
                          position probe medially in the left SFJ zone (X: 0.50–0.61) 
                          and apply Valsalva to confirm duration.",
  "clinical_reason": "Reflux detected at N1→N2; confirm with sustained maneuver.",
  "flow_type": "RP",
  "anatomical_location": "Left SFJ zone (X: 0.65, Y: 0.08)",
  "reflux_duration": "1.1s",
  "confidence": 0.88
}
```

---

## Implementation Details

### Data Flow: Single Position Analysis with Personalization

```
1. User selects sonographer (e.g., Dr. Sarah Chen)
   ↓
2. User enters ultrasound data or clicks "Analyze Previous Session"
   ↓
3. Frontend sends POST /api/probe-guidance with sonographer_id
   ↓
4. Backend:
   a. Retrieve sonographer profile from DB
   b. Retrieve last 3 sessions for this sonographer
   c. Build sonographer_context string with:
      - Name, experience, specialty
      - Known scanning style
      - Anatomical zone coordinates
      - Previous session summary (reflux patterns, durations)
   ↓
5. RAG retrieval: Query medical knowledge base
   ↓
6. Build LLM prompt with:
   - Clinical finding (flow type, location, duration)
   - Probe zone description with coordinates
   - Medical context from RAG
   - FULL sonographer profile + history
   ↓
7. LLM generates personalized guidance:
   - Adapts to sonographer's known techniques
   - References their past scanning patterns
   - Uses specific coordinate zones
   - Suggests next steps based on their style
   ↓
8. Return guidance to frontend
   ↓
9. Frontend displays with session context
```

### Data Flow: Session Analysis

```
1. User selects "Analyze Previous Session" mode
   ↓
2. UI displays cards of last 5 sessions with summaries
   ↓
3. User clicks on a session card
   ↓
4. Frontend calls handleAnalyzePreviousSession():
   a. Computes session statistics:
      - Total guidance points
      - Reflux % vs Normal %
      - Top reflux locations (sampled)
      - Estimated session duration
   ↓
5. UI displays:
   - Session overview with statistics
   - Key findings (reflux hotspots + normal areas)
   - Personalized insights (pattern analysis)
   - Complete guidance history in scrollable list
   ↓
6. Insights tell user:
   - How this sonographer scans (their style)
   - What patterns they detected
   - Recommendations for future sessions
```

---

## How to Use

### For End Users (Sonographers)

1. **Access Task-2**: Navigate to main menu → "Probe Guidance"
2. **Select Profile**: Click your name or a peer's profile from the 3 available
3. **Choose Analysis Mode**:
   - **Single Position**: Analyze one ultrasound finding
   - **Stream Multiple**: Process sequential findings in real-time
   - **Analyze Previous Session**: Review past sessions for pattern insights
4. **Receive Personalized Guidance**: LLM adapts recommendations to your style

### For Clinicians/Managers

- **Profile Review**: Open any sonographer's profile to see session count + trends
- **Pattern Analysis**: "Analyze Previous Session" reveals individual scanning habits
- **Comparative Learning**: Compare guidance across different sonographers' profiles
- **Progressive Training**: Use historical data to identify improvement areas

---

## Technical Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React.js + Axios |
| **Backend** | Flask + Python |
| **LLM** | Groq (llama-3.1-70b-versatile) |
| **RAG** | Qdrant vector database |
| **Database** | SQLite (mlops_metrics.db) |
| **Video/Imaging** | OpenCV + PyTorch |

---

## Future Enhancements

1. **Real-Time Session Saving**: Auto-save analysis sessions as they're created
2. **Comparative Analytics**: Show guidance patterns across Multiple sonographers
3. **ML-Based Pattern Recognition**: Identify sonographer-specific reflux detection rates
4. **Feedback Loop**: Rate guidance helpfulness → refine future recommendations
5. **Predictive Guidance**: ML predicts next optimal probe position based on history
6. **Cohort Benchmarking**: Compare individual sonographer metrics vs. team averages
7. **Export Reports**: Generate PDF session summaries with personalized insights

---

## Troubleshooting

### Issue: "No previous sessions" when analyzing
**Cause**: Sonographer profile exists but no sessions saved yet
**Solution**: Complete at least one stream/single analysis to create session data

### Issue: Guidance doesn't feel "personalized"
**Cause**: Sonographer context may not be strongly influencing LLM output
**Solution**: 
- Check that `sonographer_id` is being sent to backend
- Verify DB has sessions for this sonographer
- Consider increasing RAG context emphasis in backend prompt

### Issue: Anatomical coordinates seem off
**Cause**: Possible mismatch between expected zones and actual probe position
**Solution**: Verify input `posXRatio` and `posYRatio` match the coordinate key in documentation

---

## Testing Checklist

- [ ] Load SonographerProfiles page; see 3 profiles displayed
- [ ] Click one profile → navigate to ProbeGuidance page
- [ ] See "Past Sessions" panel showing recent sessions
- [ ] Switch to "Analyze Previous Session" mode
- [ ] Select a session → click "Analyze Selected Session"
- [ ] Verify session statistics display (total clips, reflux %, etc.)
- [ ] See personalized insights matching sonographer profile
- [ ] Complete new analysis → new session saved to DB
- [ ] Refresh page → new session appears in history
- [ ] Verify guidance includes zone-specific coordinate references
- [ ] Check LLM adapts language based on sonographer's experience level

