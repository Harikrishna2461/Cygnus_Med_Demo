"""
New API Endpoints for Task-1 (Temporal Flow Analysis) and Task-2 (Probe Navigation)
To be integrated into app.py
"""

# =====================
# ENDPOINT TASK-1A: Temporal Flow Analysis (Single Data Point)
# =====================
# @app.route('/api/task-1/temporal-flow', methods=['POST'])
def analyze_temporal_flow_point():
    """
    Analyze a single temporal flow data point and detect abnormal patterns.
    
    Request:
        {
            "data_point": {
                "sequenceNumber": int,
                "fromType": "N1"|"N2"|"N3"|"P"|"B",
                "toType": "N1"|"N2"|"N3"|"P"|"B",
                "step": "SFJ-Knee" or other step name,
                "flow": "EP"|"RP",
                "clipPath": str,
                "legSide": "left"|"right",
                "posXRatio": float,
                "posYRatio": float,
                ...
            },
            "analyzer_session_id": str (optional, reuse analyzer)
        }
    
    Response:
        {
            "status": "processing" | "abnormal_flow_detected" | "shunt_classified",
            "flow_summary": {...},
            "abnormal_pattern": {...} (if detected),
            "shunt_classification": {...} (if available),
            "analyzer_session_id": str (for subsequent calls)
        }
    """
    request_start = time.time()
    run_id = None
    
    try:
        data = request.json
        data_point = data.get('data_point', {})
        analyzer_session_id = data.get('analyzer_session_id')
        
        if not analyzer_session_id:
            # Create new analyzer for this session
            analyzer_session_id = str(uuid.uuid4())
        
        # Initialize or get analyzer from session
        if not hasattr(analyze_temporal_flow_point, 'analyzers'):
            analyze_temporal_flow_point.analyzers = {}
        
        if analyzer_session_id not in analyze_temporal_flow_point.analyzers:
            analyze_temporal_flow_point.analyzers[analyzer_session_id] = TemporalFlowAnalyzer()
        
        analyzer = analyze_temporal_flow_point.analyzers[analyzer_session_id]
        
        # Get or create run
        run_id, request_number, is_new_run = get_or_create_run(
            task_name='Task-1 Temporal Flow',
            task_type='stream',
            description=f'Temporal flow analysis: {data_point.get("step", "unknown")}'
        )
        
        # Add flow point and check for abnormal patterns
        abnormal = analyzer.add_flow_point(data_point)
        flow_summary = analyzer.get_flow_summary()
        
        result = {
            "status": "processing",
            "flow_summary": flow_summary,
            "analyzer_session_id": analyzer_session_id,
            "sequence_number": data_point.get('sequenceNumber', 0)
        }
        
        # If abnormal pattern detected, classify shunt
        if abnormal:
            result["status"] = "abnormal_flow_detected"
            result["abnormal_pattern"] = {
                "pattern_sequence": abnormal.pattern_sequence,
                "is_circular": abnormal.is_circular,
                "severity": abnormal.severity,
                "entry_point": abnormal.entry_point,
                "exit_point": abnormal.exit_point,
                "reflux_points": abnormal.reflux_points
            }
            
            # Get shunt classification
            shunt = analyzer.get_classified_shunt()
            if shunt:
                result["status"] = "shunt_classified"
                result["shunt_classification"] = shunt
                
                # Generate ligation plan if generator available
                if LIGATION_GENERATOR_AVAILABLE:
                    try:
                        gen = create_ligation_generator(call_llm, retrieve_context)
                        ligation_plan = gen.generate_treatment_plan(
                            shunt_type=shunt["shunt_type"],
                            flow_pattern=abnormal.pattern_sequence,
                            patient_context={},
                            reasoning=shunt["description"]
                        )
                        result["ligation_plan"] = ligation_plan
                    except Exception as e:
                        logger.warning(f"Could not generate ligation plan: {e}")
        
        # Record metrics
        elapsed_ms = (time.time() - request_start) * 1000
        mlops_tracker.record_request_metric(
            run_id=run_id,
            task_name='Task-1 Temporal Flow',
            request_number=request_number,
            metric_dict={
                'start_time': datetime.now().isoformat(),
                'end_time': datetime.now().isoformat(),
                'response_time_ms': elapsed_ms,
                'status': result["status"],
                'abnormal_detected': abnormal is not None,
                'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
                'cpu_percent': psutil.cpu_percent()
            }
        )
        
        logger.info(f"✓ Task-1: {result['status']} - Sequence {data_point.get('sequenceNumber', 0)}")
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Task-1 error: {e}")
        if run_id:
            mlops_tracker.record_request_metric(
                run_id=run_id,
                task_name='Task-1 Temporal Flow',
                request_number=1,
                metric_dict={
                    'start_time': datetime.now().isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'response_time_ms': (time.time() - request_start) * 1000,
                    'error': str(e),
                    'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
                    'cpu_percent': psutil.cpu_percent()
                }
            )
        return jsonify({"error": str(e), "status": "error"}), 500


# =====================
# ENDPOINT TASK-1B: Temporal Flow Stream (Multiple Sequential Points)  
# =====================
# @app.route('/api/task-1/temporal-flow-stream', methods=['POST'])
def analyze_temporal_flow_stream():
    """
    Analyze a stream of temporal flow data points.
    Each point represents a vein transition in sequence.
    
    Request:
        {
            "data_stream": [
                { "sequenceNumber": 1, "fromType": "N1", "toType": "N2", "step": "SFJ-Knee", ... },
                { "sequenceNumber": 2, "fromType": "N2", "toType": "N3", "step": "SFJ-Knee", ... },
                { "sequenceNumber": 3, "fromType": "N3", "toType": "N1", "step": "SFJ-Knee", ... }
            ],
            "patient_context": {
                "age": 45,
                "hemodynamic_class": "C2",
                "symptoms": "varicose veins",
                "leg_side": "right"
            }
        }
    
    Response:
        {
            "run_id": str,
            "total_processed": int,
            "flow_summary": {...},
            "detected_shunts": [
                {
                    "shunt_type": "Type 3",
                    "pattern": ["N1", "N2", "N3", "N1"],
                    "severity": "moderate",
                    "ligation_plan": {...}
                }, ...
            ],
            "clinical_recommendations": str
        }
    """
    stream_start = time.time()
    run_id = None
    
    try:
        data = request.json
        data_stream = data.get('data_stream', [])
        patient_context = data.get('patient_context', {})
        
        if not data_stream:
            return jsonify({"error": "No data stream provided"}), 400
        
        logger.info(f"Task-1 Stream: Processing {len(data_stream)} flow points")
        
        # Get or create run
        run_id, _, is_new_run = get_or_create_run(
            task_name='Task-1 Temporal Flow Stream',
            task_type='stream',
            description=f'Temporal flow stream with {len(data_stream)} points',
            num_samples=len(data_stream)
        )
        
        # Initialize analyzer
        analyzer = TemporalFlowAnalyzer()
        detected_shunts = []
        
        # Process stream
        for i, data_point in enumerate(data_stream):
            try:
                abnormal = analyzer.add_flow_point(data_point)
                
                # If shunt detected, classify it
                if abnormal:
                    shunt = analyzer.get_classified_shunt()
                    if shunt:
                        # Generate detailed ligation plan
                        ligation_plan = None
                        if LIGATION_GENERATOR_AVAILABLE:
                            try:
                                gen = create_ligation_generator(call_llm, retrieve_context)
                                ligation_plan = gen.generate_treatment_plan(
                                    shunt_type=shunt["shunt_type"],
                                    flow_pattern=abnormal.pattern_sequence,
                                    patient_context=patient_context,
                                    reasoning=shunt["description"]
                                )
                            except Exception as e:
                                logger.warning(f"Ligation plan generation failed: {e}")
                        
                        detected_shunts.append({
                            "detected_at_point": i + 1,
                            "shunt_type": shunt["shunt_type"],
                            "pattern": abnormal.pattern_sequence,
                            "severity": abnormal.severity,
                            "description": shunt["description"],
                            "reflux_type": shunt["reflux_type"],
                            "ligation_plan": ligation_plan
                        })
                
                # Record point metric
                point_elapsed_ms = (time.time() - stream_start) * 1000
                mlops_tracker.record_request_metric(
                    run_id=run_id,
                    task_name='Task-1 Temporal Flow Stream',
                    request_number=i + 1,
                    metric_dict={
                        'start_time': datetime.now().isoformat(),
                        'end_time': datetime.now().isoformat(),
                        'response_time_ms': point_elapsed_ms,
                        'abnormal_detected': abnormal is not None,
                        'sequence_number': data_point.get('sequenceNumber', 0),
                        'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
                        'cpu_percent': psutil.cpu_percent()
                    }
                )
                
            except Exception as e:
                logger.error(f"Error processing point {i}: {e}")
                mlops_tracker.record_request_metric(
                    run_id=run_id,
                    task_name='Task-1 Temporal Flow Stream',
                    request_number=i + 1,
                    metric_dict={
                        'start_time': datetime.now().isoformat(),
                        'end_time': datetime.now().isoformat(),
                        'response_time_ms': (time.time() - stream_start) * 1000,
                        'error': str(e),
                        'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
                        'cpu_percent': psutil.cpu_percent()
                    }
                )
        
        # Generate clinical summary if shunt detected
        clinical_summary = ""
        if detected_shunts:
            primary_shunt = detected_shunts[0]
            clinical_summary = (
                f"Detected {primary_shunt['shunt_type']} with severity '{primary_shunt['severity']}'. "
                f"Flow pattern: {' → '.join(primary_shunt['pattern'])}. "
                f"Classification: {primary_shunt['description']}"
            )
        
        elapsed_ms = (time.time() - stream_start) * 1000
        
        result = {
            "run_id": run_id,
            "total_processed": len(data_stream),
            "detected_shunts": detected_shunts,
            "clinical_summary": clinical_summary,
            "flow_summary": analyzer.get_flow_summary(),
            "metrics": {
                "total_duration_ms": elapsed_ms,
                "average_per_point_ms": elapsed_ms / len(data_stream) if data_stream else 0
            }
        }
        
        logger.info(f"✓ Task-1 Stream complete: {len(detected_shunts)} shunts detected")
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Task-1 Stream error: {e}")
        if run_id:
            mlops_tracker.record_request_metric(
                run_id=run_id,
                task_name='Task-1 Temporal Flow Stream',
                request_number=1,
                metric_dict={
                    'error': str(e),
                    'response_time_ms': (time.time() - stream_start) * 1000,
                    'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
                    'cpu_percent': psutil.cpu_percent()
                }
            )
        return jsonify({"error": str(e), "status": "error"}), 500


# =====================
# ENDPOINT TASK-2A: Probe Navigation (Real-time Guidance)
# =====================
# @app.route('/api/task-2/probe-navigation', methods=['POST'])
def get_probe_navigation_guidance():
    """
    Get real-time guidance for probe positioning relative to groin.
    Live processing (non-temporal) to guide surgeon towards varicose veins.
    
    Request:
        {
            "probe_position": {
                "posXRatio": float (0-1, 0=medial, 1=lateral),
                "posYRatio": float (0-1, 0=proximal/groin, 1=distal),
                "flow": "EP"|"RP",
                "legSide": "left"|"right",
                "fromType": "N1"|"N2"|"N3"|"P"|"B",
                "toType": "N1"|"N2"|"N3"|"P"|"B",
                "step": str
            },
            "target_vein": "N1"|"N2"|"N3"|"GSV"|"SSV" (optional),
            "pathology_type": "gsv_incompetence"|"ssv_incompetence"|"perforator"|"pelvic" (optional)
        }
    
    Response:
        {
            "status": "success",
            "current_location": {
                "region": str (groin, upper_thigh, mid_thigh, etc.),
                "x_ratio": float,
                "y_ratio": float,
                "anatomical_context": str
            },
            "guidance": {
                "primary_instruction": str,
                "next_action": str,
                "target_landmark": str,
                "urgency": "routine"|"important"|"critical"
            },
            "expected_veins": [str],
            "next_steps": [str]
        }
    """
    request_start = time.time()
    run_id = None
    
    try:
        data = request.json
        probe_position = data.get('probe_position', {})
        target_vein = data.get('target_vein')
        pathology_type = data.get('pathology_type')
        
        # Get or create run
        run_id, request_number, is_new_run = get_or_create_run(
            task_name='Task-2 Probe Navigation',
            task_type='single',
            description=f'Probe guidance to {target_vein or "target vein"}'
        )
        
        # Initialize navigator
        if not hasattr(get_probe_navigation_guidance, 'navigator'):
            get_probe_navigation_guidance.navigator = ProbeNavigator()
        
        navigator = get_probe_navigation_guidance.navigator
        
        # Update probe position and get guidance
        guidance_result = navigator.update_probe_position(probe_position)
        
        # Add real-time instruction
        primary_instruction = navigator.provide_real_time_guidance(probe_position)
        
        if "error" not in guidance_result:
            guidance_result["guidance"]["primary_instruction"] = primary_instruction
        
        # Record metrics
        elapsed_ms = (time.time() - request_start) * 1000
        mlops_tracker.record_request_metric(
            run_id=run_id,
            task_name='Task-2 Probe Navigation',
            request_number=request_number,
            metric_dict={
                'start_time': datetime.now().isoformat(),
                'end_time': datetime.now().isoformat(),
                'response_time_ms': elapsed_ms,
                'target_vein': target_vein or 'unknown',
                'pathology': pathology_type or 'unknown',
                'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
                'cpu_percent': psutil.cpu_percent()
            }
        )
        
        logger.info(f"✓ Task-2: Probe guidance provided for {probe_position.get('step', 'unknown')}")
        return jsonify(guidance_result), 200
        
    except Exception as e:
        logger.error(f"Task-2 error: {e}")
        if run_id:
            mlops_tracker.record_request_metric(
                run_id=run_id,
                task_name='Task-2 Probe Navigation',
                request_number=1,
                metric_dict={
                    'error': str(e),
                    'response_time_ms': (time.time() - request_start) * 1000,
                    'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
                    'cpu_percent': psutil.cpu_percent()
                }
            )
        return jsonify({"error": str(e), "status": "error"}), 500


# =====================
# ENDPOINT TASK-2B: Probe Navigation Stream (Sequential Positions)
# =====================
# @app.route('/api/task-2/probe-navigation-stream', methods=['POST'])
def get_probe_navigation_stream():
    """
    Process sequential probe positions and provide real-time surgical guidance.
    Maps the ultrasound scanning path and provides step-by-step instructions.
    
    Request:
        {
            "position_stream": [
                {
                    "sequenceNumber": 1,
                    "posXRatio": float,
                    "posYRatio": float,
                    "flow": "EP"|"RP",
                    "step": "SFJ-Knee",
                    ...
                }, ...
            ]
        }
    
    Response:
        {
            "run_id": str,
            "total_positions": int,
            "navigation_path": [
                {
                    "sequence": int,
                    "location": str,
                    "instruction": str,
                    "expected_findings": [str]
                }, ...
            ],
            "scanning_summary": str,
            "clinical_endpoints_reached": [str]
        }
    """
    stream_start = time.time()
    run_id = None
    
    try:
        data = request.json
        position_stream = data.get('position_stream', [])
        
        if not position_stream:
            return jsonify({"error": "No position stream provided"}), 400
        
        logger.info(f"Task-2 Stream: Processing {len(position_stream)} probe positions")
        
        # Get or create run
        run_id, _, is_new_run = get_or_create_run(
            task_name='Task-2 Probe Navigation Stream',
            task_type='stream',
            description=f'Probe navigation with {len(position_stream)} positions',
            num_samples=len(position_stream)
        )
        
        # Initialize navigator
        navigator = ProbeNavigator()
        navigation_path = []
        clinical_endpoints = set()
        
        # Process position stream
        for i, probe_pos in enumerate(position_stream):
            try:
                guidance = navigator.update_probe_position(probe_pos)
                
                if "error" not in guidance:
                    nav_point = {
                        "sequence": i + 1,
                        "location": guidance.get("current_location", {}).get("region", "unknown"),
                        "instruction": navigator.provide_real_time_guidance(probe_pos),
                        "expected_findings": guidance.get("expected_veins", []),
                        "actual_finding": probe_pos.get("fromType", "unknown")
                    }
                    navigation_path.append(nav_point)
                    
                    # Track clinical endpoints (EP/RP)
                    if probe_pos.get("flow") == "EP":
                        clinical_endpoints.add("Entry Point marked")
                    elif probe_pos.get("flow") == "RP":
                        clinical_endpoints.add("Re-entry Point marked")
                
                # Record stream metric
                mlops_tracker.record_request_metric(
                    run_id=run_id,
                    task_name='Task-2 Probe Navigation Stream',
                    request_number=i + 1,
                    metric_dict={
                        'start_time': datetime.now().isoformat(),
                        'end_time': datetime.now().isoformat(),
                        'response_time_ms': (time.time() - stream_start) * 1000,
                        'sequence': i + 1,
                        'location': probe_pos.get('step', 'unknown'),
                        'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
                        'cpu_percent': psutil.cpu_percent()
                    }
                )
                
            except Exception as e:
                logger.error(f"Error processing position {i}: {e}")
                mlops_tracker.record_request_metric(
                    run_id=run_id,
                    task_name='Task-2 Probe Navigation Stream',
                    request_number=i + 1,
                    metric_dict={
                        'error': str(e),
                        'response_time_ms': (time.time() - stream_start) * 1000,
                        'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
                        'cpu_percent': psutil.cpu_percent()
                    }
                )
        
        # Generate scanning summary
        regions_visited = set(p["location"] for p in navigation_path)
        scanning_summary = f"Scanned {', '.join(sorted(regions_visited))} regions with {len(navigation_path)} positions recorded"
        
        elapsed_ms = (time.time() - stream_start) * 1000
        
        result = {
            "run_id": run_id,
            "total_positions": len(position_stream),
            "navigation_path": navigation_path,
            "scanning_summary": scanning_summary,
            "clinical_endpoints_reached": list(clinical_endpoints),
            "metrics": {
                "total_duration_ms": elapsed_ms,
                "average_per_position_ms": elapsed_ms / len(position_stream) if position_stream else 0
            }
        }
        
        logger.info(f"✓ Task-2 Stream complete: {len(navigation_path)} positions processed")
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Task-2 Stream error: {e}")
        if run_id:
            mlops_tracker.record_request_metric(
                run_id=run_id,
                task_name='Task-2 Probe Navigation Stream',
                request_number=1,
                metric_dict={
                    'error': str(e),
                    'response_time_ms': (time.time() - stream_start) * 1000,
                    'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
                    'cpu_percent': psutil.cpu_percent()
                }
            )
        return jsonify({"error": str(e), "status": "error"}), 500
