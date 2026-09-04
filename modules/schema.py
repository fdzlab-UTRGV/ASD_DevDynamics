"""
schema.py — SPARK dataset instrument definitions
Mirrors the JavaScript SCHEMA from the web tool but in Python.
"""

# ── DCDQ ──────────────────────────────────────────────────────────────────────
DCDQ = {
    "label": "DCDQ",
    "prefix": "dcdq.",
    "color": "#a78bfa",
    "inverted": True,          # higher raw = better → invert before scoring
    "score_range": (1, 5),
    "domains": {
        "Gross Motor":  ["q01_throw_ball","q02_catch_ball","q03_hit_ball",
                         "q04_jump_obstacles","q05_run_fast_similar","q06_plan_motor_activity"],
        "Fine Motor":   ["q07_printing_writing_drawing_fast","q08_printing_letters_legible",
                         "q09_appropriate_tension_printing_writing","q10_cuts_pictures_shapes"],
        "Coordination": ["q11_likes_sports_motors_skills","q12_learns_new_motor_tasks",
                         "q13_quick_competent_tidying_up","q14_bull_in_china_shop","q15_fatigue_easily"],
    }
}

# ── RBS-R ─────────────────────────────────────────────────────────────────────
RBS = {
    "label": "RBS-R",
    "prefix": "rbsr.",
    "color": "#38bdf8",
    "score_range": (0, 3),
    "domains": {
        # SSC-compatible domains (for cross-dataset comparability)
        "Sensory":  ["q01_whole_body","q02_head","q03_hand_finger","q04_locomotion",
                     "q05_object_usage","q06_sensory","q07_hits_self_body",
                     "q08_hits_self_against_object","q09_hits_self_with_object",
                     "q10_bites_self","q11_pulls","q12_rubs","q13_inserts_finger","q14_skin_picking",
                     "q22_touch_tap","q23_eating"],
        "Obsessive": ["q15_arranging","q16_complete","q17_washing","q18_checking",
                      "q19_counting","q20_hoarding","q21_repeating","q24_sleep","q25_self_care",
                      "q26_travel","q27_play","q28_communication","q29_things_same_place",
                      "q30_objects","q31_becomes_upset","q32_insists_walking"],
        # SPARK native subscales
        "Stereotyped":     ["q01_whole_body","q02_head","q03_hand_finger","q04_locomotion","q05_object_usage","q06_sensory"],
        "Self-Injurious":  ["q07_hits_self_body","q08_hits_self_against_object","q09_hits_self_with_object",
                            "q10_bites_self","q11_pulls","q12_rubs","q13_inserts_finger","q14_skin_picking"],
        "Compulsive":      ["q15_arranging","q16_complete","q17_washing","q18_checking","q19_counting","q20_hoarding","q21_repeating"],
        "Ritualistic":     ["q24_sleep","q25_self_care","q26_travel","q27_play","q28_communication",
                            "q29_things_same_place","q30_objects","q31_becomes_upset","q32_insists_walking"],
    }
}

# ── SCQ ───────────────────────────────────────────────────────────────────────
# Items where NO=1 (typical behavior absent = symptom present)
SCQ_REVERSED = {
    "q01_phrases","q02_conversation","q09_expressions_appropriate",
    "q19_best_friend","q20_talk_friendly","q21_copy_you","q22_point_things",
    "q23_gestures_wanted","q24_nod_head","q25_shake_head","q26_look_directly",
    "q27_smile_back","q28_things_interested","q29_share","q30_join_enjoyment",
    "q31_comfort","q32_help_attention","q33_range_expressions","q34_copy_actions",
    "q35_make_believe","q36_same_age","q37_respond_positively",
}

SCQ = {
    "label": "SCQ",
    "prefix": "scq.",
    "color": "#4ade80",
    "score_range": (0, 1),
    "reversed_items": SCQ_REVERSED,
    "domains": {
        "Social":        ["q04_inappropriate_question","q09_expressions_appropriate",
                          "q19_best_friend","q20_talk_friendly","q21_copy_you","q26_look_directly",
                          "q27_smile_back","q28_things_interested","q29_share","q30_join_enjoyment",
                          "q31_comfort","q32_help_attention","q33_range_expressions","q34_copy_actions",
                          "q35_make_believe","q36_same_age","q37_respond_positively"],
        "Sensory":       ["q03_odd_phrase","q07_same_over","q08_particular_way","q14_senses",
                          "q15_odd_ways","q16_complicated_movements","q17_injured_deliberately","q18_objects_carry"],
        "Communication": ["q01_phrases","q02_conversation","q05_pronouns_mixed",
                          "q06_invented_words","q10_hand_tool"],
        "Unclassified":  ["q11_interest_preoccupy","q12_parts_object","q13_interests_intensity"],
    }
}

# ── ADOS (revised algorithm item mappings) ─────────────────────────────────────
# SA = Social Affect, RRB = Restricted & Repetitive Behaviors
# Scores 3→2, 7/8/9→0 per algorithm convention
ADOS_MODULES = {
    "ados_original_module_1": {
        "prefix": "ados_original_module_1.",
        "sa":    ["a2_frequency_vocalization","a3_intonation_vocalization","a4_immediate_echolalia",
                  "a5_stereotyped_words","a6_others_body","a7_pointing","a8_gestures",
                  "b1_eye_contact","b2_responsive_smile","b3_facial_expressions_others","b4_integration_gaze",
                  "b5_shared_enjoyment","b6_response_name","b7_requesting","b8_giving",
                  "b9_showing","b10_spontaneous_joint","b11_response_attention","b12_quality_overtures"],
        "rrb":   ["d1_unusual_sensory","d2_hand_complex_mannerisms","d3_self_injurious","d4_unusually_repetitive"],
        "play":  ["c1_functional_play","c2_imagination"],
        "other": ["e1_overactivity","e2_tantrums","e3_anxiety"],
    },
    "ados_original_module_2": {
        "prefix": "ados_original_module_2.",
        "sa":    ["a1_non_echoed","a2_amt_overtures","a6_conversation","a7_pointing","a8_gestures",
                  "b1_eye_contact","b2_facial_expressions_others","b3_shared_enjoyment","b4_response_name",
                  "b5_showing","b6_spontaneous_joint","b7_response_attention","b8_quality_overtures",
                  "b9_quality_response","b10_reciprocal_social","b11_quality_rapport"],
        "rrb":   ["a3_speech_abnormalities","a4_immediate_echolalia","a5_stereotyped_words",
                  "d1_unusual_sensory","d2_hand_complex_mannerisms","d3_self_injurious","d4_unusually_repetitive"],
        "play":  ["c1_functional_play","c2_imagination"],
        "other": ["e1_overactivity","e2_tantrums","e3_anxiety"],
    },
    "ados_original_module_3": {
        "prefix": "ados_original_module_3.",
        "sa":    ["a5_offers_information","a6_asks_information","a7_reporting_events","a8_conversation",
                  "a9_gestures","b1_eye_contact","b2_facial_expressions_others","b4_shared_enjoyment",
                  "b5_empathy","b6_insight","b7_quality_overtures","b8_quality_response",
                  "b9_reciprocal_social","b10_quality_rapport"],
        "rrb":   ["a1_non_echoed","a2_speech_abnormalities","a3_immediate_echololia","a4_stereotyped_words",
                  "d1_unusual_sensory","d2_hand_complex_mannerisms","d3_self_injurious",
                  "d4_excessive_interest","d5_compulsions_rituals"],
        "play":  ["c1_imagination"],
        "other": ["e1_overactivity","e2_tantrums","e3_anxiety"],
    },
    "ados_original_module_4": {
        "prefix": "ados_original_module_4.",
        "sa":    ["a8_conversation","a10_emphatic_gestures","b1_eye_contact","b2_facial_expressions_others",
                  "b5_own_affect","b6_empathy","b7_insight","b9_quality_overtures",
                  "b10_quality_response","b11_reciprocal_social","b12_quality_rapport"],
        "rrb":   ["a2_speech_abnormalities","a4_stereotyped_words","d1_unusual_sensory",
                  "d2_hand_complex_mannerisms","d3_self_injurious","d4_excessive_interest","d5_compulsions_rituals"],
        "play":  ["c1_imagination"],
        "other": ["e1_overactivity","e2_tantrums","e3_anxiety"],
    },
    "ados_2_module_1": {
        "prefix": "ados_2_module_1.",
        "sa":    ["a2_frequency_vocalization","a3_intonation_vocalization","a4_immediate_echolalia",
                  "a5_stereotyped_words","a6_others_body","a7_pointing","a8_gestures",
                  "b1_eye_contact","b2_responsive_smile","b3_facial_expressions_others","b4_integration_gaze",
                  "b5_shared_enjoyment","b6_response_name","b7_requesting","b8_giving",
                  "b9_showing","b10_spontaneous_joint","b11_response_attention","b12_quality_overtures"],
        "rrb":   ["d1_unusual_sensory","d2_hand_complex_mannerisms","d3_self_injurious","d4_unusually_repetitive"],
        "play":  ["c1_functional_play","c2_imagination"],
        "other": ["e1_overactivity","e2_tantrums","e3_anxiety"],
    },
    "ados_2_module_2": {
        "prefix": "ados_2_module_2.",
        "sa":    ["a1_non_echoed","a5_conversation","a6_pointing","a7_gestures",
                  "b1_eye_contact","b2_facial_expressions_others","b3_shared_enjoyment","b4_response_name",
                  "b5_showing","b6_spontaneous_joint","b7_response_attention","b8_quality_overtures",
                  "b9a_amt_overtures_examiner","b10_quality_response","b11_reciprocal_social","b12_quality_rapport"],
        "rrb":   ["a2_speech_abnormalities","a3_immediate_echolalia","a4_stereotyped_words",
                  "d1_unusual_sensory","d2_hand_complex_mannerisms","d3_self_injurious","d4_unusually_repetitive"],
        "play":  ["c1_functional_play","c2_imagination"],
        "other": ["e1_overactivity","e2_tantrums","e3_anxiety"],
    },
    "ados_2_module_3": {
        "prefix": "ados_2_module_3.",
        "sa":    ["a5_offers_information","a6_asks_information","a7_reporting_events","a8_conversation",
                  "a9_gestures","b1_eye_contact","b2_facial_expressions_examiner","b4_shared_enjoyment",
                  "b5_empathy","b6_insight","b7_quality_overtures","b8_amt_overtures",
                  "b9_quality_response","b10_reciprocal_social","b11_quality_rapport"],
        "rrb":   ["a1_non_echoed","a2_speech_abnormalities","a3_immediate_echololia","a4_stereotyped_words",
                  "d1_unusual_sensory","d2_hand_complex_mannerisms","d3_self_injurious",
                  "d4_excessive_interest","d5_compulsions_rituals"],
        "play":  ["c1_imagination"],
        "other": ["e1_overactivity","e2_tantrums","e3_anxiety"],
    },
    "ados_2_module_4": {
        "prefix": "ados_2_module_4.",
        "sa":    ["a8_conversation","a10_emphatic_gestures","b1_eye_contact","b2_facial_expressions_examiner",
                  "b5_own_affect","b6_empathy","b7_insight","b9_quality_overtures",
                  "b10_amt_overtures","b11_quality_response","b12_reciprocal_social","b13_quality_rapport"],
        "rrb":   ["a2_speech_abnormalities","a4_stereotyped_words","d1_unusual_sensory",
                  "d2_hand_complex_mannerisms","d3_self_injurious","d4_excessive_interest","d5_compulsions_rituals"],
        "play":  ["c1_imagination"],
        "other": ["e1_overactivity","e2_tantrums","e3_anxiety"],
    },
    "ados_2_toddler": {
        "prefix": "ados_2_toddler.",
        "sa":    ["a2_spontaneous_vocalization","a3_intonation_vocalization","a7_pointing","a8_gestures",
                  "b1_eye_contact","b4_facial_expressions_others","b5_integration_gaze","b6_shared_enjoyment",
                  "b7_response_name","b9_requesting","b10_amount_requesting","b11_giving","b12_showing",
                  "b13_spontaneous_joint","b14_response_attention","b15_quality_overtures",
                  "b17_level_engagement","b18_quality_rapport"],
        "rrb":   ["d1_unusual_sensory","d2_hand_finger_movements","d3_complex_mannerisms",
                  "d4_self_injurious","d5_unusually_repetitive"],
        "play":  ["c1_functional_play","c2_imagination","c3_imitation"],
        "other": [],
    },
}

ADOS = {
    "label": "ADOS",
    "color": "#f472b6",
    "score_range": (0, 2),
    "domains": {
        "Social Affect": "sa",
        "RRB":           "rrb",
        "Play/Imag.":    "play",
        "Other Behav.":  "other",
    },
    "modules": ADOS_MODULES,
}

# ── CBCL ──────────────────────────────────────────────────────────────────────
CBCL_MAP = {
    "cbcl_1_5": {
        "prefix": "cbcl_1_5.",
        "domains": {
            "Internalizing": "internalizing_problems_raw_score",
            "Externalizing": "externalizing_problems_raw_score",
            "Anxious/Dep.":  "anxious_depressed_raw_score",
            "Withdrawn":     "withdrawn_raw_score",
            "Somatic":       "somatic_complaints_raw_score",
            "Attention":     "attention_problems_raw_score",
            "Aggressive":    "aggressive_behavior_raw_score",
            "ODD":           "dsm5_oppositional_defiant_raw_score",
            "Sleep":         "sleep_problems_raw_score",
            "Emot. React.":  "emotionally_reactive_raw_score",
        }
    },
    "cbcl_6_18": {
        "prefix": "cbcl_6_18.",
        "domains": {
            "Internalizing": "internalizing_problems_raw_score",
            "Externalizing": "externalizing_problems_raw_score",
            "Anxious/Dep.":  "anxious_depressed_raw_score",
            "Withdrawn":     "withdrawn_raw_score",
            "Somatic":       "somatic_complaints_raw_score",
            "Attention":     "attention_problems_raw_score",
            "Aggressive":    "aggressive_behavior_raw_score",
            "Rule-Breaking": "rule_breaking_raw_score",
            "Social Prob.":  "social_problems_raw_score",
            "Thought Prob.": "thought_problems_raw_score",
            "Conduct":       "dsm5_conduct_problems_raw_score",
            "ADHD":          "dsm5_attention_deficit_hyperactivity_raw_score",
            "ODD":           "dsm5_oppositional_defiant_raw_score",
        }
    }
}

CBCL = {
    "label": "CBCL",
    "color": "#fbbf24",
    "score_range": (0, 30),
    "priority": "cbcl_6_18",   # 6-18 over 1-5 when both exist
    "forms": CBCL_MAP,
    "domains": {
        # All possible unified domains (None for age-inappropriate form)
        "Internalizing": None, "Externalizing": None,
        "Anxious/Dep.": None, "Withdrawn": None, "Somatic": None,
        "Attention": None, "Aggressive": None, "ODD": None,
        "Rule-Breaking": None,   # 6-18 only
        "Social Prob.": None,    # 6-18 only
        "Thought Prob.": None,   # 6-18 only
        "Conduct": None,         # 6-18 only
        "ADHD": None,            # 6-18 only
        "Sleep": None,           # 1-5 only
        "Emot. React.": None,    # 1-5 only
    }
}

# ── COVARIATE FIELDS ──────────────────────────────────────────────────────────
COVARIATE_FIELDS = {
    # Demographics
    "sex":                  ["core_descriptive_variables.sex", "sex"],
    "age_months":           ["core_descriptive_variables.age_at_registration_months",
                             "iq.age_test_date_months", "age_at_eval_months"],
    "age_years":            ["core_descriptive_variables.age_at_registration_years",
                             "age_at_eval_years"],
    "fsiq":                 ["core_descriptive_variables.fsiq", "iq.fsiq_score",
                             "iq.fsiq", "fsiq"],
    "nviq":                 ["core_descriptive_variables.nviq", "iq.nviq_score",
                             "iq.nviq", "nviq"],
    "viq":                  ["core_descriptive_variables.viq", "iq.viq_score",
                             "iq.viq", "viq"],
    "asd":                  ["core_descriptive_variables.asd", "asd"],
    "asd_confirmed":        ["core_descriptive_variables.asd_diagnosis_confirmed"],
    "scq_total":            ["core_descriptive_variables.scq_total_final_score"],
    "rbsr_total":           ["core_descriptive_variables.rbsr_total_final_score"],
    "language_level":       ["core_descriptive_variables.language_level_latest"],
    "cognitive_impairment": ["core_descriptive_variables.cognitive_impairment_latest"],
    "ados_version":         ["core_descriptive_variables.ados_version"],
    # Developmental milestones (continuous — months)
    "walk_months":          ["core_descriptive_variables.walked_age_mos"],
    "first_words_months":   ["core_descriptive_variables.used_words_age_mos"],
    "onset_months":         ["core_descriptive_variables.age_onset_mos"],
    "diagnosis_age_months": ["core_descriptive_variables.diagnosis_age"],
    # Developmental milestones (binary)
    "language_regression":  ["core_descriptive_variables.regress_lang_y_n"],
    "other_regression":     ["core_descriptive_variables.regress_other_y_n"],
}

NUMERIC_COVARIATES = {
    "age_months", "age_years", "fsiq", "nviq", "viq",
    "scq_total", "rbsr_total",
    "walk_months", "first_words_months", "onset_months", "diagnosis_age_months",
}

# Milestone column clean labels for display
MILESTONE_LABELS = {
    "walk_months":          "Walked independently (months)",
    "first_words_months":   "First words (months)",
    "onset_months":         "Symptom onset (months)",
    "diagnosis_age_months": "Age at ASD diagnosis (months)",
    "language_regression":  "Language regression (binary)",
    "other_regression":     "Other skill regression (binary)",
}

# Sensitivity stratification options
SENSITIVITY_STRAT_OPTIONS = [
    {"label": "Sex",                    "value": "sex"},
    {"label": "ADOS module",            "value": "_ados_module"},
    {"label": "Language level",         "value": "language_level"},
    {"label": "Cognitive impairment",   "value": "cognitive_impairment"},
    {"label": "ASD confirmed only",     "value": "asd_confirmed"},
    {"label": "Age band",               "value": "_age_band"},
]

# ── SCHEMA REGISTRY ───────────────────────────────────────────────────────────
SCHEMA = {
    "dcdq": DCDQ,
    "rbs":  RBS,
    "scq":  SCQ,
    "ados": ADOS,
    "cbcl": CBCL,
}

SCALE_ORDER = ["dcdq", "rbs", "scq", "ados", "cbcl"]

# Correlation analysis defaults
CORR_PREDICTORS = [
    ("dcdq", "Gross Motor"), ("dcdq", "Fine Motor"), ("dcdq", "Coordination"),
    ("rbs",  "Sensory"),     ("rbs",  "Obsessive"),
    ("ados", "RRB"),         ("ados", "Social Affect"),
]
CORR_OUTCOMES = [
    ("scq",  "Social"),        ("scq",  "Sensory"),       ("scq",  "Communication"),
    ("ados", "Social Affect"), ("ados", "RRB"),
    ("cbcl", "Internalizing"), ("cbcl", "Externalizing"),
    ("cbcl", "Anxious/Dep."),  ("cbcl", "Social Prob."),  ("cbcl", "Attention"),
]
