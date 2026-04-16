# Task 2: Quick Reference Guide

## 🎯 System Overview

Your personalized sonographer guidance system is now **fully operational** with:

1. **Positional Zone System** - Exact anatomical coordinates for probe positioning
2. **Digital Twin Profiles** - 3 pre-configured sonographer personas that learn from history
3. **Adaptive LLM Guidance** - AI adapts recommendations based on individual sonographer style
4. **Session Persistence** - Historical data stored and used for future guidance

---

## 📍 Positional Information Reference

### Coordinate System
- **Origin**: (0, 0) = top-left corner of image
- **Extent**: (1, 1) = bottom-right corner of image
- **All positions**: Normalized to 0-1 range

### Anatomical Zones (Right & Left Legs)

#### RIGHT LEG (left side of screen)
| Zone | X Range | Y Range |
|------|---------|---------|
| **SFJ-Knee** | 0.0931–0.475 | 0–0.5497 |
| **Knee-Ankle** | 0.105–0.2947 | 0.5497–1.0 |
| **SPJ-Ankle** | 0.2827–0.4386 | 0.5497–1.0 |

#### LEFT LEG (right side of screen)
| Zone | X Range | Y Range |
|------|---------|---------|
| **SFJ-Knee** | 0.4985–0.909 | 0–0.5497 |
| **Knee-Ankle** | 0.7081–0.91 | 0.5497–1.0 |
| **SPJ-Ankle** | 0.588–0.714 | 0.5497–1.0 |

**Example**: Position X=0.65, Y=0.25 on left leg → Left thigh, Hunterian canal region

---

## 👥 Sonographer Profiles

### Profile 1: Dr. Sarah Chen
```
ID: sono-001
Experience: 12 years (SENIOR)
Specialty: Complex CHIVA & bilateral lower limb
Style: TOP-DOWN approach
  • Starts at SFJ (groin) scanning distally to knee
  • Prefers LONGITUDINAL views at SFJ
  • Uses Valsalva maneuver routinely
  • Revisits reflux points TWICE to confirm
  • Scans RIGHT leg first, then mirrors LEFT

Personalization: Get detailed, confirmatory guidance with specific SFJ positioning
```

### Profile 2: Dr. James Okoye
```
ID: sono-002
Experience: 4 years (JUNIOR)
Specialty: Type 2 perforator mapping & calf vein assessment
Style: BOTTOM-UP approach
  • Starts at ANKLE perforators (Cockett zone)
  • Works PROXIMALLY toward groin
  • Thorough calf perforator mapping
  • Occasionally misses Hunterian perforators
  • Frequent COMPRESSION maneuvers in calf zone

Personalization: Get step-by-step, simpler guidance starting from ankle upward
```

### Profile 3: Dr. Maria Santos
```
ID: sono-003
Experience: 8 years (MID-LEVEL)
Specialty: Pelvic origin reflux, SSV & SPJ assessment
Style: BILATERAL SIMULTANEOUS approach
  • Compares BOTH legs IN PARALLEL
  • Starts POSTERIORLY at SPJ/popliteal fossa
  • Expert at complex Type 4/5 patterns
  • Applies ELIMINATION TEST routinely
  • Careful SSV documentation

Personalization: Get comprehensive bilateral assessment guidance with pattern comparisons
```

---

## 🚀 How to Use

### Step 1: Navigate to Task-2
```
Main Menu → Probe Guidance (or /probe)
```

### Step 2: Select a Sonographer
```
Choose from 3 profiles:
├─ Dr. Sarah Chen (Senior, top-down systematic)
├─ Dr. James Okoye (Junior, bottom-up detailed)
└─ Dr. Maria Santos (Expert, bilateral comparison)
```

### Step 3: Choose Analysis Mode

#### **Option A: Single Position**
```
Best for: Quick guidance on one ultrasound finding

Steps:
1. Enter ultrasound data (JSON)
2. Click "🎯 Get Guidance"
3. Receive personalized probe instruction

Example Output:
"Given Dr. Sarah's preference for longitudinal views,
 move probe medially in left SFJ zone (X: 0.50–0.61)
 and apply Valsalva to confirm duration."
```

#### **Option B: Stream Multiple Positions**
```
Best for: Real-time guidance during full examination

Steps:
1. Paste sequence of ultrasound findings (JSON array)
2. Set buffer interval (default 0.5s)
3. Click "▶️ Start Stream Analysis"
4. Watch real-time guidance update for each position
5. Session auto-saved with full guidance history

Output: Live guidance + analysis statistics
```

#### **Option C: Analyze Previous Session** ⭐ NEW
```
Best for: Review past scanning patterns and learn

Steps:
1. Click "📊 Analyze Previous Session" mode
2. Select a session from cards showing:
   • Date & time
   • Total clips analyzed
   • Reflux detections
   • Session summary
3. Click "🔍 Analyze Selected Session"
4. See detailed breakdown:
   • Session statistics (clip count, reflux %)
   • Key findings (top reflux locations)
   • Personalized insights (this sonographer's style)
   • Complete guidance history

Output: Comprehensive session analysis + pattern insights
```

---

## 📊 Understanding the Guidance

### Standard Guidance Format

**Reflux Detected (RP):**
```
"Given Dr. Sarah's scanning style, move probe medially 
 in the left SFJ zone (X: 0.50–0.61, Y: 0.05–0.15) 
 and apply Valsalva maneuver to confirm reflux duration."

Key elements:
• Sonographer name → personalization
• Action verb → clear instruction
• Anatomical zone → precise location
• Zone coordinates → exact positioning
• Clinical maneuver → specific technique
```

**Normal Flow (EP):**
```
"Continue scanning GSV distally from current SFJ-Knee 
 zone along medial thigh, using longitudinal approach 
 as you typically do, to assess competence at knee level."

Key elements:
• Flow status → next steps
• Anatomical progression → where to scan next
• Personalization → their preferred approach
```

---

## 💾 Database & Session Persistence

### What Gets Saved
```
For each analysis session:
✓ Sonographer profile (ID, name, specialty)
✓ Session date/time
✓ Analysis mode (single, stream, analyze)
✓ Total clips/positions analyzed
✓ Reflux detection count
✓ Complete guidance history (flow type + instructions)
✓ Session summary (auto-generated)
```

### How It's Used
```
Next time this sonographer logs in:
1. System retrieves their profile
2. Loads last 3 sessions from database
3. Builds personalized context with:
   • Known scanning style
   • Past scanning patterns
   • Common reflux locations they detect
   • Anatomical zone coordinates
4. LLM uses this context for better guidance
5. Cycle repeats → digital twin improves
```

---

## 🔄 Personalization Pipeline

```
Request Flow:
┌─────────────────────────────────────────┐
│  Sonographer Selection + Ultrasound Data │
└────────────────┬────────────────────────┘
                 ↓
        ┌────────────────────┐
        │  Query Database    │
        │ • Profile info     │
        │ • Last 3 sessions  │
        │ • Scanning style   │
        └────────────┬───────┘
                     ↓
        ┌─────────────────────────────────┐
        │  Build Context String:          │
        │ • Name + experience             │
        │ • Scanning style description    │
        │ • Zone coordinates              │
        │ • Previous findings             │
        └────────────┬────────────────────┘
                     ↓
        ┌──────────────────────────┐
        │  LLM Prompt Creation:    │
        │ • Clinical data          │
        │ • Medical context (RAG)  │
        │ • Personalization block  │
        └────────────┬─────────────┘
                     ↓
        ┌──────────────────────────┐
        │  LLM Response:           │
        │ "Given this sonographer, │
        │  in this zone,           │
        │  with their style,       │
        │  do this..."             │
        └────────────┬─────────────┘
                     ↓
        ┌──────────────────────────┐
        │  Delivery to Sonographer │
        │ Personalized guidance!   │
        └──────────────────────────┘
```

---

## 📋 Ultrasound Data Format

### Single Position Format
```json
{
  "sequenceNumber": 1,
  "flow": "RP",              // EP (normal) or RP (reflux)
  "step": "SFJ-Knee",        // anatomical zone
  "fromType": "N1",          // vein network from
  "toType": "N2",            // vein network to
  "reflux_duration": 1.1,    // seconds
  "posXRatio": 0.65,         // X coordinate (0-1)
  "posYRatio": 0.08,         // Y coordinate (0-1)
  "legSide": "left",         // left or right
  "confidence": 0.88,        // 0-1 confidence
  "clipPath": "frame-001.png",
  "description": "Left SFJ reflux detected at junction"
}
```

### Stream Format (Array)
```json
[
  { /* point 1 */ },
  { /* point 2 */ },
  { /* point 3 */ },
  // ... more points
]
```

---

## 🎓 Example Sessions

### Session 1: Dr. Sarah Chen (Senior, Systematic)
```
Total Clips: 28
Reflux Detections: 7 (25%)
Duration: ~3.2 minutes

Pattern Analysis:
• Started at left SFJ (groin) as expected
• Traced GSV longitudinally to knee
• Applied Valsalva at 7 points (her routine)
• Revisited reflux points twice each
• Mirrored pattern on right leg
• Identified Type 1 + Type 3 patterns

Guidance Style Received:
→ Detailed, with confirmatory steps
→ SFJ-focused recommendations
→ Valsalva timing emphasized
→ Bilateral comparison suggested
```

### Session 2: Dr. James Okoye (Junior, Bottom-Up)
```
Total Clips: 35
Reflux Detections: 12 (34%)
Duration: ~4.1 minutes

Pattern Analysis:
• Started at ankle perforators (his style)
• Worked proximally through calf
• Used compression maneuver in knee-ankle zone
• Spent significant time on calf perforator mapping
• Less coverage at SFJ level initially
• Missed some Hunterian perforators (known pattern)

Guidance Style Received:
→ Step-by-step from ankle upward
→ Calf-focused recommendations first
→ Compression technique reminders
→ SFJ encouragement once in zone
```

### Session 3: Dr. Maria Santos (Expert, Bilateral)
```
Total Clips: 42
Reflux Detections: 8 (19%)
Duration: ~4.8 minutes

Pattern Analysis:
• Assessed both legs simultaneously
• Started posteriorly at SPJ each time
• Applied elimination test at points
• Identified complex Type 4/5 patterns
• Careful posterior/SPJ documentation
• Complete bilateral comparison before conclusion

Guidance Style Received:
→ Bilateral recommendation phrasing
→ Comparison-based instructions
→ Complex pattern anticipation
→ Elimination test encouragement
```

---

## ✅ Testing Commands

### Run All Tests
```bash
cd /Users/HariKrishnaD/Downloads/NUS/Internships/Cygnus/cmed_demo
python3 test_task2_personalization.py
```

### Verify Database
```bash
python3 -c "
import sys; sys.path.insert(0, 'backend')
import sonographer_db
sonographer_db.init_db()
sono_list = sonographer_db.get_all_sonographers()
print(f'Profiles: {len(sono_list)}')
for s in sono_list:
    print(f'  - {s[\"name\"]}')
"
```

---

## 🐛 Troubleshooting

### "No previous sessions" message
**Cause**: First time using this profile  
**Fix**: Complete 1-2 analyses to build session history

### Guidance doesn't seem personalized
**Cause**: Missing sonographer_id in request  
**Fix**: Ensure you selected a profile before submitting

### Zone coordinates seem off
**Cause**: Input coordinates outside expected ranges  
**Fix**: Verify posXRatio and posYRatio are 0-1 range

---

## 📞 Support Resources

| Resource | Location |
|----------|----------|
| **Full Documentation** | [TASK2_PERSONALIZED_GUIDANCE.md](TASK2_PERSONALIZED_GUIDANCE.md) |
| **Implementation Guide** | [TASK2_IMPLEMENTATION_COMPLETE.md](TASK2_IMPLEMENTATION_COMPLETE.md) |
| **Test Suite** | [test_task2_personalization.py](test_task2_personalization.py) |
| **Backend Code** | [backend/sonographer_db.py](backend/sonographer_db.py) |
| **Backend Routes** | [backend/app.py](backend/app.py) → /api/probe-guidance |
| **Frontend Component** | [frontend/src/pages/ProbeGuidance.js](frontend/src/pages/ProbeGuidance.js) |

---

## ✨ Key Features Summary

✅ **Anatomical Understanding** - System knows exact zones with coordinates  
✅ **Digital Twin Learning** - Remembers and adapts to each sonographer  
✅ **Personalized Guidance** - AI recommendations match individual style  
✅ **Session History** - All sessions saved for future reference  
✅ **Zone-Aware Navigation** - Guidance includes specific coordinate ranges  
✅ **Multi-Mode Analysis** - Single, Stream, and Session Review modes  
✅ **LLM Integration** - RAG-enhanced with personalization context  
✅ **Production Ready** - Tested, documented, and ready to deploy  

---

## 🎯 Next Steps

1. **Start using**: Open Task-2 → select a Profile → choose analysis mode
2. **Build history**: Complete multiple sessions to strengthen personalization
3. **Review patterns**: Use "Analyze Previous Session" for insights
4. **Feedback**: Rate guidance quality to help system learn
5. **Deploy**: Ready for production use!

---

**Status**: ✅ COMPLETE & TESTED  
**Last Updated**: 2026-04-15  
**Ready for**: Deployment / QA / Production

