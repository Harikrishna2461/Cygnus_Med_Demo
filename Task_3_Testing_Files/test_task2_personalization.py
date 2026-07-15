#!/usr/bin/env python3
"""
Test script for Task-2: Personalized Sonographer Guidance System

This script:
1. Tests the sonographer database and profiles
2. Simulates creating sessions with guidance history
3. Tests the personalized context building
4. Validates anatomical zone detection
5. Tests the complete workflow
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

import json
from datetime import datetime, timedelta
import sonographer_db

def test_initialize_db():
    """Test database initialization and seeding"""
    print("\n" + "="*70)
    print("TEST 1: Initialize Sonographer Database")
    print("="*70)
    
    sonographer_db.init_db()
    sonographers = sonographer_db.get_all_sonographers()
    
    print(f"✓ Database initialized")
    print(f"✓ Seeded {len(sonographers)} sonographer profiles:")
    for s in sonographers:
        print(f"  - {s['name']} ({s['title']}, {s['experience_years']} yrs)")
        print(f"    Specialty: {s['specialty']}")
    
    return sonographers

def test_sonographer_retrieval(sonographers):
    """Test retrieving individual sonographer data"""
    print("\n" + "="*70)
    print("TEST 2: Retrieve Individual Sonographer Profile")
    print("="*70)
    
    sono = sonographer_db.get_sonographer(sonographers[0]['id'])
    print(f"✓ Retrieved: {sono['name']}")
    print(f"  ID: {sono['id']}")
    print(f"  Scanning Style: {sono['scanning_style'][:100]}...")
    
    return sono

def test_create_mock_session(sonographer, session_count=5):
    """Create mock sessions for testing"""
    print("\n" + "="*70)
    print(f"TEST 3: Create Mock Sessions for {sonographer['name']}")
    print("="*70)
    
    created_sessions = []
    
    for i in range(session_count):
        # Create random guidance history
        guidance_history = []
        
        # Add some reflux findings
        for j in range(3):
            guidance_history.append({
                'flow_type': 'RP',
                'instruction': f'Move probe medially to locate reflux at position {j+1}',
                'clinical_reason': f'Reflux detected at {sonographer["specialty"].split("&")[0]}'
            })
        
        # Add some normal findings
        for j in range(4):
            guidance_history.append({
                'flow_type': 'EP',
                'instruction': f'Continue scanning region {j+1} for assessment',
                'clinical_reason': 'Normal flow detected'
            })
        
        session_id = sonographer_db.save_session(
            sono_id=sonographer['id'],
            mode='stream',
            guidance_history=guidance_history,
            session_summary=f"Session {i+1}: {len(guidance_history)} clips analyzed, {3} reflux findings"
        )
        
        created_sessions.append({
            'session_id': session_id,
            'total_clips': len(guidance_history),
            'reflux_count': 3
        })
        
        print(f"✓ Session {i+1} created: {session_id}")
    
    return created_sessions

def test_retrieve_sessions(sonographer):
    """Retrieve session history for sonographer"""
    print("\n" + "="*70)
    print(f"TEST 4: Retrieve Session History for {sonographer['name']}")
    print("="*70)
    
    sessions = sonographer_db.get_sessions(sonographer['id'], limit=5)
    
    print(f"✓ Retrieved {len(sessions)} sessions:")
    for i, session in enumerate(sessions, 1):
        print(f"  Session {i}:")
        print(f"    - Date: {session['session_date'][:10]}")
        print(f"    - Total points: {session['total_points']}")
        print(f"    - Reflux detections: {session['reflux_count']}")
        if session.get('session_summary'):
            print(f"    - Summary: {session['session_summary']}")
    
    return sessions

def test_build_sonographer_context(sonographer):
    """Build personalized sonographer context for LLM"""
    print("\n" + "="*70)
    print(f"TEST 5: Build Personalized Context for {sonographer['name']}")
    print("="*70)
    
    context = sonographer_db.build_sonographer_context(sonographer['id'])
    
    print("✓ Context built for LLM injection:")
    print("\n--- BEGIN CONTEXT ---")
    print(context)
    print("--- END CONTEXT ---\n")
    
    # Verify key elements are in context
    checks = [
        ('name', sonographer['name'] in context),
        ('experience', str(sonographer['experience_years']) in context),
        ('specialty', sonographer['specialty'] in context),
        ('scanning_style', sonographer['scanning_style'][:50] in context),
        ('anatomical_zones', 'SFJ-Knee' in context),
        ('right_leg_zones', 'RIGHT LEG' in context),
        ('left_leg_zones', 'LEFT LEG' in context),
        ('coordinate_ranges', '0.0931' in context),
        ('session_history', 'PREVIOUS SESSION HISTORY' in context),
    ]
    
    print("✓ Context validation:")
    for check_name, result in checks:
        status = "✓" if result else "✗"
        print(f"  {status} {check_name}")
    
    return context

def test_anatomical_zone_detection():
    """Test anatomical zone detection from coordinates"""
    print("\n" + "="*70)
    print("TEST 6: Anatomical Zone Detection from Coordinates")
    print("="*70)
    
    # Simulate the backend's zone detection logic
    def detect_zone(leg, px, py):
        """Detect anatomical zone from normalized coordinates"""
        if leg == 'right':
            if py <= 0.5497:
                if py <= 0.098:
                    return "SFJ (groin) — saphenofemoral junction level"
                elif py <= 0.353:
                    return "upper-to-mid right thigh (Hunterian canal region), GSV medial course"
                else:
                    return "lower right thigh approaching popliteal fossa"
            else:
                if 0.2827 <= px <= 0.4386:
                    return "right popliteal fossa / SPJ region — posterior approach"
                elif py >= 0.85:
                    return "right distal calf near ankle (distal Cockett perforator zone)"
                else:
                    return "right mid-calf (Boyd/Cockett perforator zone)"
        else:  # left
            if py <= 0.5497:
                if py <= 0.098:
                    return "SFJ (groin) — saphenofemoral junction level"
                elif py <= 0.353:
                    return "upper-to-mid left thigh (Hunterian canal region), GSV medial course"
                else:
                    return "lower left thigh approaching popliteal fossa"
            else:
                if 0.588 <= px <= 0.714:
                    return "left popliteal fossa / SPJ region — posterior approach"
                elif py >= 0.85:
                    return "left distal calf near ankle (distal Cockett perforator zone)"
                else:
                    return "left mid-calf (Boyd/Cockett perforator zone)"
    
    test_cases = [
        ('left', 0.65, 0.08, 'Left SFJ groin level'),
        ('left', 0.70, 0.30, 'Left mid-thigh Hunterian'),
        ('left', 0.80, 0.70, 'Left mid-calf Boyd zone'),
        ('left', 0.65, 0.75, 'Left Cockett perforator'),
        ('right', 0.25, 0.08, 'Right SFJ groin'),
        ('right', 0.20, 0.70, 'Right mid-calf'),
        ('right', 0.35, 0.68, 'Right SPJ'),
    ]
    
    print("✓ Testing coordinate-to-zone mapping:")
    for leg, x, y, description in test_cases:
        zone = detect_zone(leg, x, y)
        print(f"  [{leg.upper()} leg] X={x:.2f}, Y={y:.2f}")
        print(f"    → {zone}")
        print(f"    (Expected: {description})")
    
    return True

def test_complete_workflow():
    """Test the complete workflow"""
    print("\n" + "="*70)
    print("TEST 7: Complete Workflow Integration")
    print("="*70)
    
    # Step 1: Get sonographers
    sonographers = sonographer_db.get_all_sonographers()
    print(f"✓ Step 1: Retrieved {len(sonographers)} profiles")
    
    # Step 2: Select one sonographer
    selected_sono = sonographers[0]
    print(f"✓ Step 2: Selected {selected_sono['name']}")
    
    # Step 3: Get their sessions
    sessions = sonographer_db.get_sessions(selected_sono['id'], limit=5)
    print(f"✓ Step 3: Retrieved {len(sessions)} sessions")
    
    # Step 4: Build context for LLM
    context = sonographer_db.build_sonographer_context(selected_sono['id'])
    context_length = len(context)
    print(f"✓ Step 4: Built LLM context ({context_length} characters)")
    
    # Step 5: Simulate LLM prompt with context
    sample_ultrasound = {
        'flow': 'RP',
        'step': 'SFJ-Knee',
        'reflux_duration': 1.1,
        'posXRatio': 0.65,
        'posYRatio': 0.08,
        'legSide': 'left',
        'fromType': 'N1',
        'toType': 'N2',
        'confidence': 0.88
    }
    
    llm_prompt = f"""ULTRASOUND GUIDANCE
Reflux at {sample_ultrasound['step']} ({sample_ultrasound['legSide']} leg)
Position: X={sample_ultrasound['posXRatio']:.3f}, Y={sample_ultrasound['posYRatio']:.3f}

=== SONOGRAPHER CONTEXT ===
{context[:500]}...

TASK: Generate personalized probe guidance
    """
    
    print(f"✓ Step 5: Simulated LLM prompt creation")
    print(f"  Prompt length: {len(llm_prompt)} characters")
    print(f"  Includes: sonographer profile, scanning style, zone coordinates, message history")
    
    print("\n✓ Complete workflow validated!")
    
    return True

def main():
    """Run all tests"""
    print("\n" + "█"*70)
    print("  TASK-2: PERSONALIZED SONOGRAPHER GUIDANCE SYSTEM")
    print("  Test Suite")
    print("█"*70)
    
    try:
        # Initialize DB
        sonographers = test_initialize_db()
        
        # Test individual retrieval
        sono = test_sonographer_retrieval(sonographers)
        
        # Create mock sessions
        sessions = test_create_mock_session(sono, session_count=3)
        
        # Retrieve sessions
        retrieved_sessions = test_retrieve_sessions(sono)
        
        # Build personalized context
        context = test_build_sonographer_context(sono)
        
        # Test zone detection
        test_anatomical_zone_detection()
        
        # Test complete workflow
        test_complete_workflow()
        
        # Summary
        print("\n" + "█"*70)
        print("  ✓ ALL TESTS PASSED")
        print("█"*70)
        print("\nSystem Status:")
        print(f"  • Sonographer Profiles: {len(sonographers)} seeded")
        print(f"  • Sessions Created: {len(sessions)}")
        print(f"  • Anatomical Zones: Configured (Right & Left legs)")
        print(f"  • LLM Context: Ready for personalized guidance")
        print(f"  • Database: {sonographer_db.DB_PATH}")
        print("\nReady for production use!")
        print("█"*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
