
# Define pretty names for reported variables
prettyNames <- c(
  # ------- Variables drawn from dispatch center
  "disp_age"= "Dispatch - Patient Age",
  "disp_gender"= "Dispatch - Patient Gender", # 0 = male, 1 = female
  "disp_lastContactN"= "Dispatch - Number of prior contacts (30 days)",
  "disp_lastContactHours"= "Dispatch - Hours since last contact", # 721 (30*24+1) if never or over 30 days
  "disp_cat"= "Dispatch - CDSS category", # One hot encoded, can have multiple
  "disp_q"= "Dispatch - CDSS questions", # NA if not available in category, otherwise encoded as described in the manuscript
  "disp_nqs"= "Dispatch - Number of CDSS questions answered",
  "disp_recpout"= "Dispatch - CDSS recommended priority",
  "disp_hour" = "Dispatch - Hour of call",
  "disp_month" = "Dispatch - Month of call",
  "disp_date" = "Dispatch - Date of call",
  "disp_disttoed"= "Dispatch - Distance to nearest ED", # In km "as the crow flies"
  
  "ambj"= "Ambulance Journal exists",
  "amb_pout" = "Ambulance - Priority to scene", # 4 = 1A, 3 = 1B, 2 = 2A, 1 = 2B. Technically, this comes from the dispatch center, but is only available as a predictor in the ambulance dataset
  # -------- Interventions by ambulance crews 0 = not performed, 1 = performed, unless otherwise noted
  "amb_txp"= "Ambulance - Patient transported",
  "amb_int_iv"= "Ambulance - IV placed",
  "amb_meds"= "Ambulance - Medications administered",
  "amb_int_immob"= "Ambulance - Patient immobilized",
  "amb_int_cpr"= "Ambulance - CPR administered", 
  "amb_int_o2"= "Ambulance - Oxygen administered (LPM)", # 0 to 25 liters per minute
  "amb_int_ekg"= "Ambulance - 12-lead EKG taken/sent to CICU", # 1 = taken, 2 = sent
  "amb_int_alert"= "Ambulance - Pre-arrival notification given",
  "amb_int_pin"= "Ambulance - Emergent transport to hospital",
  "amb_any"= "Ambulance - Any intervention provided", # 1 if any of the above are present
  
  # ------ Clinical findings by ambulance crews - One hot encoded, can have multiple findings unless otherwise noted
  "amb_calltypes"= "Ambulance - Call types",
  "amb_crit"= "Ambulance - Critical patient status", # 0 = not critical, 1 = critical
  "amb_airway"= "Ambulance - Airway findings",
  "amb_breathing"= "Ambulance - Breathing findings",
  "amb_circ"= "Ambulance - Circulation findings",
  "amb_pulse"= "Ambulance - Pulse quality",
  "amb_skincond"= "Ambulance - Skin condition",
  "amb_breathsounds"= "Ambulance - Breathing sounds",
  "amb_ptmeds"= "Ambulance - Patient medications",
  "amb_medhist"= "Ambulance - Patient medical history",
  
  # ------ Time intervals in minutes
  "amb_time_disp"= "Ambulance - Time to dispatch", # call recieved to unit en route
  "amb_time_scene"= "Ambulance - Time on scene", # unit on scene to unit en route to hospital
  "amb_time_toed"= "Ambulance - Time to hospital", # unit en route to hospital to unit arrived at hospital
  
  # ------ First documented vital signs by ambulance crews (average if documented by multiple ambulances on scene)
  "amb_v_br"= "Ambulance - Respiration rate",
  "amb_v_spo2"= "Ambulance - SpO2",
  "amb_v_pr"= "Ambulance - Pulse rate",
  "amb_v_bp"= "Ambulance - Systolic blood pressure",
  "amb_v_temp"= "Ambulance - Temperature",
  "amb_v_avpu"= "Ambulance - AVPU", # 4 = Alert, 3 = Verbal, 2 = Pain, 1 = Unconscious
  "amb_v_gcs"= "Ambulance - GCS",
  
  "hospj"= "Hospital record exists",
  # ------- Hospital outcomes
  "hosp_admit"= "Hospital admission",
  "hosp_ed"= "Treatment at ED",
  "hosp_icu"= "Treatment at Intensive Care Unit",
  "hosp_deadinhosp"= "In-hospital mortality",
  "hosp_critcare"= "Critical Care",
  "hosp_dead2day"= "Two-day mortality",
  "hosp_triage_single" = "ED triage category",
  
  # ------- Study exclusion criteria (0 = criteria not met, 1 = criteria met)
  "incl_pin"= "Missing PIN",
  "incl_ambj"= "No ambulance journal",
  "incl_ambtxp"= "No ambulance transport",
  "incl_hospj"= "No hospital journal",
  "incl_ed"= "No ED visit",
  "incl_vitals"= "Missing > 2 vitals", 
  "incl_adult"= "Patient age < 18", 
  "incl_disp"= "No dispatch CDSS data", 
  "incl_final"= "Call included", 
  "incl_hospnontxp" = "Transport with no hospital journal",
  
  # ------- Other variables used in analysis, transformations, or reporting results 
  "CreatedOn"= "Call received timestamp",
  "pin"= "Has valid Personal Identification Number",
  "coord_n"= "Coordinate (Lat)",
  "coord_e"= "Coordinate (Lon)",
  "pout"= "Dispatched priority",
  "news"= "NEWS Score", 
  "amb_composite" = "Ambulance risk score", 
  "disp_composite" = "Dispatch risk score", 
  "pub_composite" = "Public risk score",
  "nmissvitals"= "Number of missing vitals",
  "var" = "Exclusion critera",
  "N_excluded" = "Excluded, N",
  "N_remain" = "Remaining, N",
  "remainpct_excluded" = "Remaining calls excluded, percent",
  "test" = "Call included in test set (2018)", # 0 = training set, 1 = test set
  "pct_train" = "Training set, percent (95% CI)",
  "pct_test" = "Test set, percent (95% CI)",
  "pct_excl" = "Excluded, percent",
  "valtype" = "Validation method",
  "lab" = "Outcome",
  "n_vars" = "Number of variables from group included in any ambulance model",
  "pct_value" = "Percent of included calls with non-zero/non-missing value",
  "n_value" = "Number of included calls with non-zero/non-missing value",
  "sum_gain" = "Average gain from inclusion of variables in ambulance models",
  
  # ------- Translate  dispatch categories
  "disp_cat_Trauma" = "Dispatch - Trauma", 
  "disp_cat_Brostsmarta" = "Dispatch - Chest pain", 
  "disp_cat_Andningsbesvar" = "Dispatch - Difficulty Breathing",
  "disp_cat_Stroke" = "Dispatch - Stroke", 
  "disp_cat_Buk...flanksmarta" = "Dispatch - Abdominal pain",
  "disp_cat_Allergisk.reaktion" = "Dispatch - Allergic reaction", 
  "disp_cat_Allman.aldring" = "Dispatch - General elderly",
  "disp_cat_Allman.barn" = "Dispatch - General pediatric",
  "disp_cat_Allman.vuxen" = "Dispatch - General adult",
  "disp_cat_Arm...bensymtom..ej.trauma." = "Dispatch - Arm/leg symptoms", 
  "disp_cat_Blod.i.urin" = "Dispatch - Blood in urine", 
  "disp_cat_Blodig.upphostning" = "Dispatch - Blood in sputum", 
  "disp_cat_Blodsocker.hogt." = "Dispatch - High blood sugar", 
  "disp_cat_Blodsocker.lagt" = "Dispatch - Low blood sugar", 
  "disp_cat_Brannskada" = "Dispatch - Burns", 
  "disp_cat_Diarre" = "Dispatch - Diarrhea", 
  "disp_cat_Drunkningstillbud" = "Dispatch - Drowning", 
  "disp_cat_Dykeriolycka" = "Dispatch - Diving accident", 
  "disp_cat_Elektrisk.skada" = "Dispatch - Electrical injury", 
  "disp_cat_Feber" = "Dispatch - Fever", 
  "disp_cat_Forlossning" = "Dispatch - Childbirth", 
  "disp_cat_Forvirring" = "Dispatch - Confusion", 
  "disp_cat_Fratskada" = "Dispatch - Chemical burn", 
  "disp_cat_Graviditet" = "Dispatch - Pregnancy", 
  "disp_cat_Hallucination" = "Dispatch - Hallucination", 
  "disp_cat_Halsont" = "Dispatch - Throat pain", 
  "disp_cat_Hjartstopp" = "Dispatch - Caridac arrest", 
  "disp_cat_Huvudvark" = "Dispatch - Headache", 
  "disp_cat_Hypertermi" = "Dispatch - Hyperthermia", 
  "disp_cat_Hypotermi" = "Dispatch - Hypothermia", 
  "disp_cat_ICD" = "Dispatch - ICD issue", 
  "disp_cat_Illamaende" = "Dispatch - Nausea", 
  "disp_cat_Infektion" = "Dispatch - Infection", 
  "disp_cat_Intox.forgiftning" = "Dispatch - Intox/poisoning", 
  "disp_cat_Kemisk.exponering" = "Dispatch - Chemical exposure", 
  "disp_cat_Koldskada" = "Dispatch - Cold injury", 
  "disp_cat_Krakning" = "Dispatch - Vomiting", 
  "disp_cat_Kramper" = "Dispatch - Seizure", 
  "disp_cat_Luftvagsbesvar" = "Dispatch - Airway problem", 
  "disp_cat_Mag...tarmblodning" = "Dispatch - GI bleed", 
  "disp_cat_Nas...svalgblodning" = "Dispatch - nosebleed", 
  "disp_cat_ogon" = "Dispatch - Eye problem", 
  "disp_cat_Ormbett" = "Dispatch - Snakebite", 
  "disp_cat_Pacemaker" = "Dispatch - Pacemaker issue", 
  "disp_cat_Psykiska.besvar" = "Dispatch - Psychiatric issue", 
  "disp_cat_Rokexponering" = "Dispatch - Smoke exposure", 
  "disp_cat_Ryggsmarta" = "Dispatch - Back pain", 
  "disp_cat_Rytmrubbning" = "Dispatch - Arrythmia", 
  "disp_cat_Sankt.vakenhet" = "Dispatch - Reduced consciousnedd", 
  "disp_cat_Sarskada" = "Dispatch - Minor trauma", 
  "disp_cat_Sensoriskt...motoriskt.bortfall" = "Dispatch - Sensory/motor loss", 
  "disp_cat_Svimning" = "Dispatch - Fainting", 
  "disp_cat_Talpaverkan" = "Dispatch - Speech issue", 
  "disp_cat_Urinkateterstopp" = "Dispatch - Urinary catheter blockage", 
  "disp_cat_Urinstamma" = "Dispatch - Dysuria", 
  "disp_cat_Urogenitala.besvar" = "Dispatch - Urogenital issue", 
  "disp_cat_Vaginal.blodning" = "Dispatch - Vaginal bleed", 
  "disp_cat_Varmeslag" = "Dispatch - Heat stroke", 
  "disp_cat_Yrsel" = "Dispatch - Dizziness", 
  
  "amb_meds_Adrenalin.0.1mg.ml." = "Meds - Adrenaline (0.1mg/ml)", 
  "amb_meds_Adrenalin.1mg.ml." = "Meds - Adrenaline (1mg/ml)", 
  "amb_meds_Alvedon.250mg." = "Meds - Paracetamol (250mg)", 
  "amb_meds_Alvedon.500mg." = "Meds - Paracetamol (500mg)", 
  "amb_meds_Alvedonsupp.125mg." = "Meds - Paracetamol supp. (125mg)", 
  "amb_meds_Atropin.0.5mg.ml." = "Meds - Atropine (0.5mg/ml)", 
  "amb_meds_Betapred.0.5mg." = "Meds - Betapred (0.5mg)", 
  "amb_meds_Bricanyl.0.5mg.ml" = "Meds - Bricanyl (0.5mg/ml)", 
  "amb_meds_Brilique.90mg.tabl." = "Meds - Brillique (90mg)", 
  "amb_meds_Carbomix.25g." = "Meds - Carbomix (25g)", 
  "amb_meds_Cordarone.50mg.ml." = "Meds - Cordarone (50mg/ml)", 
  "amb_meds_EMLAplaster.25mg." = "Meds - EMLA patch (25mg)", 
  "amb_meds_Furix.10mg.ml." = "Meds - Furix (10mg/ml)", 
  "amb_meds_Glukos.300mg.ml." = "Meds - Glucose (300mg/ml)", 
  "amb_meds_Glukos.inf.50mg.ml." = "Meds - Glucose infused (50mg/ml)", 
  "amb_meds_Glytrin.0.4mg.dos." = "Meds - Glytrin (50mg/ml)", 
  "amb_meds_Heparin.5000E.ml." = "Meds - Heparin (5000E/ml)", 
  "amb_meds_Ketamin.10mg.ml." = "Meds - Ketamine (10mg/ml)", 
  "amb_meds_Lanexat.0.1mg.ml." = "Meds - Lanexat (0.1mg/ml)", 
  "amb_meds_Medimix.50.N2O.50.O2." = "Meds - Medimix (50/50)", 
  "amb_meds_Midazolam.1mg.ml." = "Meds - Midazolam (1mg/ml)", 
  "amb_meds_Midazolam.5mg.ml." = "Meds - Midazolam (5mg/ml)", 
  "amb_meds_Morfin.10mg.ml." = "Meds - Morphine (10mg/ml)", 
  "amb_meds_Naloxon.0.4mg.ml." = "Meds - Naloxone (0.4mg/ml)", 
  "amb_meds_Natriumklorid.9mg.ml." = "Meds - Sodium chloride (9mg/ml)", 
  "amb_meds_Ondansetron.2mg.ml." = "Meds - Ondansetron (2mg/ml)", 
  "amb_meds_ovrigtLakemedel" = "Meds - Other medication", 
  "amb_meds_Ringer.Acetat" = "Meds - Ringers acetate", 
  "amb_meds_Seloken.1mg.ml." = "Meds - Metoprolol (1mg/ml)", 
  "amb_meds_Solu.Cortef.50mg.ml." = "Meds - Hydrocoritsone (50mg/ml)", 
  "amb_meds_Stesolid.5mg.ml." = "Meds - Diazepam (5mg/ml)", 
  "amb_meds_Stesolid.klysma.10mg." = "Meds - Diazepam supp. (10mg/ml)", 
  "amb_meds_Stesolid.klysma.5mg." = "Meds - Diazepam supp (5mg/ml)", 
  "amb_meds_Suscard.2.5mg.tabl." = "Meds - Nitroglycerine (2.5mg tab)", 
  "amb_meds_Tapinplaster.25mg." = "Meds - Lidocaine patch (25mg)", 
  "amb_meds_Tavegyl.1mg.ml." = "Meds - Clemastine (1mg/ml)", 
  "amb_meds_Trombyl.75mg.tabl." = "Meds - Aspirin (75mg tab)", 
  "amb_meds_Ventoline.2mg.ml." = "Meds - Ventoline (2mg/ml)", 
  "amb_meds_Voltaren.25mg.ml." = "Meds - Diclofenac (25mg/ml)", 
  "amb_meds_Xylocain.1..spray" = "Meds - Xylocaine (1 spray)", 
  
  "amb_calltypes_Allergiskreaktion" = "Ambulance - Allergic reaction", 
  "amb_calltypes_Andningsbesvar" = "Ambulance - Difficulty breathing", 
  "amb_calltypes_Arm.bensymtomejtrauma" = "Ambulance - Arm/leg symptoms", 
  "amb_calltypes_Avliden" = "Ambulance - Dead on arrival", 
  "amb_calltypes_Brannskada" = "Ambulance - Burns", 
  "amb_calltypes_Brostsmarta" = "Ambulance - Chest pain", 
  "amb_calltypes_Buk.flanksmarta" = "Ambulance - Abdominal pain", 
  "amb_calltypes_Drunkningstillbud" = "Ambulance - Drowning", 
  "amb_calltypes_Ejrelevant" = "Ambulance - Not relevant", 
  "amb_calltypes_Elektriskskada" = "Ambulance - Electircal burn", 
  "amb_calltypes_Forlossning.Graviditet" = "Ambulance - Childbirth/pregnancy", 
  "amb_calltypes_Forvirring" = "Ambulance - Confusion", 
  "amb_calltypes_Hjartstopp" = "Ambulance - Cardiac arrest", 
  "amb_calltypes_Huvudvark" = "Ambulance - Headache", 
  "amb_calltypes_Hypo..Hyperglykemi" = "Ambulance - Hypo-/Hyperglycemia", 
  "amb_calltypes_Hypotermi.Hypertermi" = "Ambulance - Hypo-/Hyperthermia", 
  "amb_calltypes_Illamaende.Krakning" = "Ambulance - Nausea/Vomiting", 
  "amb_calltypes_Infektion.Feber" = "Ambulance - Infection/fever", 
  "amb_calltypes_Intox.Forgiftning" = "Ambulance - Intox/poisoning", 
  "amb_calltypes_Kramper" = "Ambulance - Seizure", 
  "amb_calltypes_Mag.tarmblodning" = "Ambulance - GI Bleed", 
  "amb_calltypes_Nas.svalgblodning" = "Ambulance - Nosebleed", 
  "amb_calltypes_Psykiskabesvar" = "Ambulance - Psychiatric issue", 
  "amb_calltypes_Ryggsmarta" = "Ambulance - Back pain", 
  "amb_calltypes_Rytmrubbning" = "Ambulance - Arrythmia", 
  "amb_calltypes_Sanktvakenhet" = "Ambulance - Reduced consciousness", 
  "amb_calltypes_Sarskada" = "Ambulance - Minor trauma", 
  "amb_calltypes_Seannatsymtom" = "Ambulance - Other symptoms", 
  "amb_calltypes_Sepsismisstanke" = "Ambulance - Suspected sepsis", 
  "amb_calltypes_Strokemisstanke" = "Ambulance - Suspected stroke", 
  "amb_calltypes_Suicid.Suicidforsok" = "Ambulance - Suicide attempt", 
  "amb_calltypes_Svimning" = "Ambulance - Fainting", 
  "amb_calltypes_Trauma" = "Ambulance - Trauma", 
  "amb_calltypes_Urinkateterstopp" = "Ambulance - Urinary catheter block", 
  "amb_calltypes_Uro.Genitalablodningar" = "Ambulance - Urogenital bleed", 
  "amb_calltypes_Urogenitalabesvar" = "Ambulance - Urogenital issue", 
  "amb_calltypes_Yrsel" = "Ambulance - Nausea", 
  
  "amb_airway_Fri" = "Airway - Free", 
  "amb_airway_Ofri" = "Airway - Blocked", 
  "amb_airway_Paverkad" = "Airway - Threatened", 
  
  "amb_breathing_Anstrangd" = "Breathing - Laboured", 
  "amb_breathing_Forhojd" = "Breathing - Raised", 
  "amb_breathing_Langsam" = "Breathing - Slow", 
  "amb_breathing_Normal" = "Breathing - Normal", 
  "amb_breathing_Okoord..stotig" = "Breathing - Agonal", 
  "amb_breathing_Saknas" = "Breathing - None", 
  "amb_breathing_Snabb" = "Breathing - Rapid", 
  "amb_breathing_Ytlig" = "Breathing - Shallow", 
  
  "amb_circ_Bradykard" = "Circulation - Bradycardic", 
  "amb_circ_Normal" = "Circulation - Normal", 
  "amb_circ_Saknas" = "Circulation - None", 
  "amb_circ_Takykard" = "Circulation - Tachycardic", 
  
  "amb_pulse_Bradykard" = "Pulse - Slow", 
  "amb_pulse_Normal" = "Pulse - Normal", 
  "amb_pulse_Oregelbunden" = "Pulse - Irregular", 
  "amb_pulse_Regelbunden" = "Pulse - Regular", 
  "amb_pulse_Saknas" = "Pulse - None", 
  "amb_pulse_Takykard" = "Pulse - Fast", 
  "amb_pulse_Tunn" = "Pulse - Weak", 
  "amb_pulse_Valfylld" = "Pulse - Strong", 
  
  "amb_skincond_Blek" = "Skin - Pale", 
  "amb_skincond_Cyanotisk" = "Skin - Cyanotic", 
  "amb_skincond_Normal" = "Skin - Normal", 
  "amb_skincond_Rodnad" = "Skin - Flushed", 
  
  "amb_breathsounds_Normal" = "Sounds - Normal", 
  "amb_breathsounds_Saknas" = "Sounds - None", 
  "amb_breathsounds_Snarkande.Gurglande" = "Sounds - Snoring/Gurgling", 
  "amb_breathsounds_Stridor.Vasande" = "Sounds - Stridor", 
  
  "amb_ptmeds_Acetylsalicylsyra" = "Pt. Meds - Aspirin", 
  "amb_ptmeds_Analgetika" = "Pt. Meds - Analgesics", 
  "amb_ptmeds_Antiarytmika" = "Pt. Meds - Antiarrythmics", 
  "amb_ptmeds_Antibiotika" = "Pt. Meds - Antibiotics", 
  "amb_ptmeds_Antidepressiva" = "Pt. Meds - Antidepressants", 
  "amb_ptmeds_Antiepileptika" = "Pt. Meds - Antiepileptics", 
  "amb_ptmeds_Antihypertensiva" = "Pt. Meds - Anithypertensive", 
  "amb_ptmeds_ASA" = "Pt. Meds - ASA", 
  "amb_ptmeds_Bensodiazepiner" = "Pt. Meds - Benzodiazepams", 
  "amb_ptmeds_Betablockerare" = "Pt. Meds - Beta blockers", 
  "amb_ptmeds_Blodfortunnande" = "Pt. Meds - Blood thinners", 
  "amb_ptmeds_Bronkdliaterande" = "Pt. Meds - Bronchodialators", 
  "amb_ptmeds_Diuretikum" = "Pt. Meds - Diuretics", 
  "amb_ptmeds_Heparin.typ" = "Pt. Meds - Hepatin-type", 
  "amb_ptmeds_Insulin" = "Pt. Meds - Insulin", 
  "amb_ptmeds_Kortison" = "Pt. Meds - Cortisone", 
  "amb_ptmeds_Lipidsankande" = "Pt. Meds - Lipid-lowering", 
  "amb_ptmeds_Nitroglycerin" = "Pt. Meds - Nitroglycerin", 
  "amb_ptmeds_NOAK.typ" = "Pt. Meds - Oral anticoagulants", 
  "amb_ptmeds_NSAID.typ" = "Pt. Meds - NSAIDs", 
  "amb_ptmeds_opioid.typ" = "Pt. Meds - Opioids", 
  "amb_ptmeds_opioidtyp" = "Pt. Meds - Opioids", 
  "amb_ptmeds_Oxygen" = "Pt. Meds - Oxygen", 
  "amb_ptmeds_paracetamol.typ" = "Pt. Meds - Paracetamol", 
  "amb_ptmeds_Peroralantidiabetikum" = "Pt. Meds - Oral antidiabetics", 
  "amb_ptmeds_Warfarin.typ" = "Pt. Meds - Warfarin-type", 
  
  "amb_medhist_Diabetes" = "Med hist. - Diabetes", 
  "amb_medhist_Hjart.karlsjd" = "Med hist. - Heart disease", 
  "amb_medhist_Langvarigsmarta" = "Med hist. - Chrinic pain", 
  "amb_medhist_Malignitet" = "Med hist. - Malignent cancer", 
  "amb_medhist_Neurologisksjd" = "Med hist. - Neurological disease", 
  "amb_medhist_Psykiatrisksjd" = "Med hist. - Psychiatric disease", 
  "amb_medhist_Respirationssjd" = "Med hist. - Resporatory disease",
  
  # ------- ... And the most common ED triage categories for Supplementary analysis 8
  "Allergisk reaktion" = "Allergic Reaction", 
  "Allman svaghet" = "General weakness", 
  "Andningsbesvar" = "Difficulty Breathing", 
  "Axelskada" = "Shoulder injury", 
  "Brostsmarta" = "Chest pain", 
  "Buksmarta" = "Abdominal pain", 
  "Extremitetssvullnad/vark" = "Swelling/pain in extremity", 
  "Feber" = "Fever", 
  "Forgiftning" = "Poisoning", 
  "Fotleds-/fotskada" = "Foot/ankle injury", 
  "Hoftskada" = "Hip injury", 
  "Huvudskada" = "Head injury", 
  "Huvudvark" = "Headache", 
  "Kramper" = "Cramping", 
  "Mag-tarmblodning" = "Abdominal bleed", 
  "Neurologiska besvar" = "Neurological difficulty", 
  "Ryggvark" = "Back pain", 
  "Rytmrubbning" = "Arrythmia", 
  "Sankt vakenhet" = "Reduced consciousness", 
  "Sarskada" = "Minor trauma", 
  "Svimning" = "Fainting", 
  "Trauma" = "Major trauma", 
  "Yrsel" = "Dizziness",
  
  # ------- some diagnostic test abbreviations
  "se" = "Sensitivity", 
  "sp" = "Specificity", 
  "diag.acc" = "Accuracy", 
  "diag.or" = "Odds ratio", 
  "ppv" = "Pos. Pred. Value", 
  "npv" = "Neg. Pred. Value")

news_calc <- function(data){
  
  # Calculates a NEWS score from vital signs
  # From https://www.rcplondon.ac.uk/file/9434/download?token=kf8WbPib
  
  out <- data %>%
    mutate(news_rr = ifelse(eval_breaths <= 8, 3,
                            ifelse(eval_breaths <= 11,1,
                                   ifelse(eval_breaths <= 20,0,
                                          ifelse(eval_breaths <= 24,2, 3)))),
           # We'll use scale 1 of the NEWS scoring chart
           news_spo2 = ifelse(eval_spo2 <= 91, 3,
                              ifelse(eval_spo2 <= 93,2,
                                     ifelse(eval_spo2 <= 95,1, 0))),
           news_o2 = ifelse(amb_o2 > 0,2,0),
           news_sbp = ifelse(eval_sbp <= 90, 3,
                             ifelse(eval_sbp <= 100,2,
                                    ifelse(eval_sbp <= 110,1,
                                           ifelse(eval_sbp <= 219,0, 3)))),
           news_pr = ifelse(eval_pulse <= 40, 3,
                            ifelse(eval_pulse <= 50,1,
                                   ifelse(eval_pulse <= 90,0,
                                          ifelse(eval_pulse <= 110,1,
                                                 ifelse(eval_pulse <= 130,2, 3))))),
           
           news_con = ifelse(eval_avpu == "(A) Alert",0,3),
           
           news_temp = ifelse(eval_temp <= 35, 3,
                              ifelse(eval_temp <= 36,1,
                                     ifelse(eval_temp <= 38,0,
                                            ifelse(eval_temp <= 39,1, 2))))) %>%
    mutate(news_full = rowSums(dplyr::select(., starts_with("news_"))))
  return(out)
}

cip_calc <- function(data){
  
  # Calculates a CIP score from age and vital signs
  # From Table 2, https://jamanetwork.com/journals/jama/article-abstract/186400
  
  out <- data %>%
    mutate(cip_age = ifelse(disp_age >= 45, 1, 0),
           cip_sbp = ifelse(amb_v_bp <= 90, 1, 0),
           cip_pr = ifelse(amb_v_pr >= 120, 1,0),
           cip_br = ifelse(amb_v_br >= 36,2,
                           ifelse(amb_v_br >= 24,1,0)),
           cip_sp02 = ifelse(amb_v_spo2 <= 87, 1,0),
           cip_gcs = ifelse(amb_v_gcs <8,2,
                            ifelse(amb_v_gcs <= 14,1,0))) %>%
    mutate(cip_full = rowSums(dplyr::select(., starts_with("cip_"))))
  return(out)
}


paste_prop_se <- function(d, r = 1){
  
  # Paste with arithmetic standard errors for proportions
  
  phat = mean(d,na.rm = T)
  n = sum(!is.na(d))
  return(paste0(round(phat*100,r),
                " ±",
                round(sqrt((phat*(1-phat))/n)*100,r)))
}

paste_mean_se <- function(d, r = 1){
  
  # Paste with arithmetic standard errors for normal distributions
  
  phat = mean(d,na.rm = T)
  n = sum(!is.na(d))
  return(paste0(round(phat,r),
                " ±",
                round(phat/sqrt(n),r)))
}



meanfun <- function(data, i){
  
  # bootstrap function for a mean value
  
  d <- data[i]
  return(mean(d,na.rm = T))   
}



boot_ci <- function(d, p = T, func = meanfun){
  
  # Boostrap 95% confidence intervals (for means by default) 
  
  bo <- boot(d, statistic=func, R=1000)
  bci <- boot.ci(bo, conf=0.95, type = "perc")
  
  if(p){
    bci$t0 <- bci$t0 * 100
    bci$percent <- bci$percent * 100
  }
  
  return(list(val = bci$t0,
              lcl = bci$percent[4],
              ucl = bci$percent[5]))
  
}



paste_boot_ci <- function(d, r = 1,prop = T, f = meanfun, newline = T){
  
  # Print boostrapped CIs (for means by default) 
  
  bci <- boot_ci(d, p = prop, func = f)
  
  return(paste0(format(round(bci$val,r),nsmall = r),
                ifelse(newline,"\n("," ("),
                format(round(bci$lcl,r),nsmall = r),
                "-",
                format(round(bci$ucl,r),nsmall = r),
                ")"))
}

desc_table <- function(x){
  
  # Generate data to populate table of descriptive statistics
  
  dplyr::summarise(x, 
            N = n(),
            'Age, mean' = paste_boot_ci(disp_age, prop = F),
            'Female, percent' = paste_boot_ci(disp_gender),
            'Emergent transport ,\npercent' = paste_boot_ci(amb_int_pin),
            'Ambulance intervention*,\n percent' = paste_boot_ci(amb_any),
            'Missing vitals,\npercent' = paste_boot_ci(ifelse(nmissvitals>0,1,0)),
            'NEWS value,\nmean' = paste_boot_ci(news,r = 2, prop = F),
            'Prior contacts\n(30 days), mean' = paste_boot_ci(disp_lastContactN,r = 2, prop = F),
            'Intensive Care Unit, percent' = paste_boot_ci(hosp_icu),
            'In-hospital death,\npercent' = paste_boot_ci(hosp_deadinhosp),
            'Critical care,\npercent' = paste_boot_ci(hosp_critcare),
            'Admitted, percent' = paste_boot_ci(hosp_admit),
            '2-day mortality,\npercent' = paste_boot_ci(hosp_dead2day))
}


generate_rocdata <- function(preds,lab){
  
  # Generate data to base a ROC curve plot on
  
  rocfun <- function(p,l){
    out <- roc.curve(p[l==1],
                     p[l==0],curve = T)$curve
    return(out)
  }
  
  vals <- rbind(lapply(preds,rocfun,lab))
  prednames <- rep(colnames(preds), times = unlist(lapply(vals,nrow)))
  
  out <- do.call(rbind,vals)
  out <- data.frame(prednames,out,type = "roc",stringsAsFactors = F)
  colnames(out) <- c("predictor","x","y","threshold","type")
  return(out)
}


generate_prdata <- function(preds,lab){
  
  # Generate data to base a Precsion/Recall curve plot on
  
  prfun <- function(p,l){
    out <- pr.curve(p[l==1],
                    p[l==0],curve = T)$curve
    return(out)
  }
  
  vals <- rbind(lapply(preds,prfun,lab))
  prednames <- rep(colnames(preds), times = unlist(lapply(vals,nrow)))
  
  out <- do.call(rbind,vals)
  out <- data.frame(prednames,out,type = "pr",stringsAsFactors = F)
  colnames(out) <- c("predictor","x","y","threshold","type")
  return(out)
}

generate_dca <- function(xgb_disp_preds,xgb_amb_preds,lab, full = incldata){
  
  # Generate data to base decision curve analysis plots on plot on
  
  preds <- data.frame(label = lab,
                      prio = 5-as.numeric(full$pout),
                      news = full$news,
                      xgb_disp = xgb_disp_preds,
                      xgb_amb = xgb_amb_preds)
  
  prio <- decision_curve(label ~ prio,
                         data = preds, 
                         study.design = "cohort", 
                         policy = "opt-out",
                         bootstraps = 50)
  
  news <- decision_curve(label ~ news,
                         data = preds, 
                         study.design = "cohort", 
                         policy = "opt-out",
                         bootstraps = 50)
  
  xgb_disp <- decision_curve(label ~ xgb_disp,
                             data = preds, 
                             study.design = "cohort", 
                             policy = "opt-out",
                             bootstraps = 50)
  
  xgb_amb <- decision_curve(label ~ xgb_amb,
                            data = preds, 
                            study.design = "cohort", 
                            policy = "opt-out",
                            bootstraps = 50)
  
  return(list(prio, news, xgb_disp, xgb_amb))
}

generate_synthdata <- function(d,n){
  
  out <- if(is.na(d["vals"])){
    # If no frequency table, generate data with a normal distribution truncated to some lower and upper limits
    round(trunc_rnorm(n,
                      d["Mean"][[1]],
                      d["sd"][[1]],
                      d["Min."][[1]],
                      d["Max."][[1]]),
          digits = d["n_decimals"][[1]])
  }else{
    # If there is a frequency table, generate values based on those frequencies
    sample(rep(as.numeric(names(d["vals"][[1]])),d["vals"][[1]]))
  }
  
  return(out)
}

trimvals <- function(x,q = c(0,0.9995)){
  if(!is.numeric(x)) return(x)
  trims <- quantile(x,q,na.rm = T)
  out <- ifelse(x >= trims[1] & x <= trims[2],x,NA)
  return(out)
}

boot_auc <- function(d,l, r = 2, type = "roc", boot = T,newline = T, print = T){
  
  # Bootstrap Area Under the Curve values using the PRROC package
  
  rocaucfun <- function(data, label, i){
    d <- data[i]
    l <- label[i]
    auc <- roc.curve(d[l==1],
                     d[l==0])
    return(auc$auc)   
  }
  
  praucfun <- function(data, label, i){
    d <- data[i]
    l <- label[i]
    auc <- pr.curve(d[l==1],
                    d[l==0])
    return(auc$auc.integral)   
  }
  
  bootfun <- switch(type,
                    "roc" = rocaucfun,
                    "pr" = praucfun)
  
  bo <- boot(d, statistic=bootfun, R=1000, label = l, strata = l)
  
  bci <- boot.ci(bo, conf=0.95, type = "perc")
  if(is.null(bci)){
    bci = list()
    bci$t0 = NA
    bci$percent = rep(NA,5)
  }
  
  if(print == T){
    return(paste0(format(round(bci$t0,r),nsmall = r),
                  if(boot){
                    paste0(ifelse(newline,"\n("," ("),
                           format(round(bci$percent[4],r),nsmall = r),
                           "-",
                           format(round(bci$percent[5],r),nsmall = r),
                           ")")}))
  }else{
    return(bci)
  }
  
}

generate_composite <- function(preds,weights,normalize = T,log_trans = T){
  
  # Generate a composite risk score from model predictions
  
  pre_trans <- NA
  
  if(normalize){
    # Perform pre-compositing normalization if desired
    preds <- scale(preds)
    # Save the scaling attributes for later
    pre_trans  <- list("center" = attr(preds, "scaled:center"), 
                       "scale" = attr(preds, "scaled:scale"))
  } 
  
  # Weight and combine predictions
  preds <- rowSums( preds %*% diag(weights))
  
  m <- NA
  
  if(log_trans){
    # Perform a log transform if desired
    
    m <- min(preds)
    
    preds <- scale(log(preds-min(preds)+0.1)) # Note we add a small offset here to ensure that there are no zero values
    
  }else{
    preds <- scale(preds)
  }
  # Save post-compositing scaling attributes
  post_trans <- list("center" = attr(preds, "scaled:center"),
                     "scale" = attr(preds, "scaled:scale"))
  
  # Add scaling attributes to returned object
  attr(preds,"pre_trans") <- pre_trans
  attr(preds,"post_trans") <- post_trans
  attr(preds,"post_min") <- m
  
  return(preds)
}

predict_composite <- function(preds,
                              cvpreds,
                              weights,
                              normalize = T,
                              log_trans = T,
                              use_min = T){
  
  # Generate new predictions based on model predictions and scaling attributes from previously generated predictions
  
  pre_trans <- NA
  
  if(normalize){
    # Scale predictions based on previously used attributes if desited
    p <- scale(preds,
                   attr(cvpreds,"pre_trans")$center,
                   attr(cvpreds,"pre_trans")$scale)
  } 
  
  # Weight and composite predictions
  pw <-rowSums(p %*% diag(weights))
  
  m <- NA
  if(use_min == T){
    m <- attr(cvpreds,"post_min")
  }else{
    m <- min(pw)
  }
  
  if(log_trans){
    
    # If applying function to predict scores using the same set of weights 
    # as used to generate the cvpreds object, use the minimum value. 
    # Otherwise, the minimum value of the set of new predictions can be used.
    # Using the minimum value of the new predictions won't work for single
    # predictions though.
    
    
    pout <- scale(log(pw-m+0.1), # Apply log transform and scale
                   attr(cvpreds,"post_trans")$center,
                   attr(cvpreds,"post_trans")$scale)
  }else{
    # Only scale
    pout <- scale(pw,
                   attr(cvpreds,"post_trans")$center,
                   attr(cvpreds,"post_trans")$scale)
  }
  
  return(pout)
}

cv_gridsearch <- function(label_tune,data_tune,params){
  
  # Perform a grid serach using a provided paramtere list
  
  performance <- apply(params, 1, function(parameterList,label_apply,data_apply){
    
    startTime <- Sys.time()
    
    xgboostModelCV <- xgb.cv(data = data_apply[[parameterList[["data"]]]],
                             label = label_apply[[parameterList[["label"]]]],
                             nrounds = 400, 
                             nfold = 5,  
                             verbose = TRUE,
                             eval.metric = parameterList[["eval.metric"]],
                             objective = parameterList[["objective"]],
                             max.depth = parameterList[["max_depth"]], 
                             eta = parameterList[["eta"]], 
                             gamma = parameterList[["gamma"]], 
                             subsample = parameterList[["subsample"]], 
                             colsample_bytree = parameterList[["colsample_bytree"]],
                             print_every_n = 10,
                             min_child_weight = parameterList[["min_child"]],
                             early_stopping_rounds = 10)
    
    # Save model performance in final round of data
    
    xvalidationScores <- as.data.frame(xgboostModelCV$evaluation_log)
    auc <- tail(xvalidationScores$test_auc_mean, 1)
    tauc <- tail(xvalidationScores$train_auc_mean,1)
    nrounds <- max(xvalidationScores$iter)
    
    output <- return(c("test_auc" = auc, 
                       "train_auc" = tauc, 
                       "time" = Sys.time() - startTime,
                       "nrounds" = nrounds,
                       parameterList))
  },label_apply = label_tune,data_apply = data_tune)
  
  # Return a dataframe containing grid search results
  
  output <- as.data.frame(unlist(t(performance),recursive = F),stringsAsFactors = F)
  return(output)
}

top_gridsearch <- function(ll,dl,sg,n){
  cv_gridsearch(label_tune = lablist,
               data_tune = datlist,
               params = sg) %>%
    
    # Save best performing hyperparameters for each model
    
    group_by(data,label) %>%
    mutate_at(vars(-label,-data,-eval.metric,-objective),as.numeric) %>%
    top_n(n = n, wt = test_auc) %>%
    ungroup()
}

cv_xgb_trainmodels <- function(label_train,data_train,params){
  
  # Train crosss-validated models
  
  models <- lapply(split(params,seq(nrow(params))), function(parameterList,label_apply,data_apply){
    
    xgboostModelCV <- xgb.cv(data = data_apply[[parameterList[["data"]]]],
                             label = label_apply[[parameterList[["label"]]]],
                             nrounds = parameterList[["nrounds"]], 
                             nfold = 5,  
                             verbose = TRUE,
                             eval.metric = parameterList[["eval.metric"]],
                             objective = parameterList[["objective"]],
                             max.depth = parameterList[["max_depth"]], 
                             eta = parameterList[["eta"]], 
                             gamma = parameterList[["gamma"]], 
                             subsample = parameterList[["subsample"]], 
                             colsample_bytree = parameterList[["colsample_bytree"]],
                             print_every_n = 10,
                             min_child_weight = parameterList[["min_child"]],
                             prediction = T)
    
    return(xgboostModelCV)
    
  }, label_apply = label_train, data_apply = data_train)
  
  names(models) <- paste(params$data,params$label,sep = "_")
  
  return(models)
}

xgb_trainmodels <- function(label_train,data_train,params){
  
  # Train non-cross validated models
  
  models <- lapply(split(params,seq(nrow(params))), function(parameterList,label_apply,data_apply){
    
    xgboostModel <- xgboost(data = data_apply[[parameterList[["data"]]]],
                            label = label_apply[[parameterList[["label"]]]],
                            nrounds = parameterList[["nrounds"]],
                            verbose = TRUE,
                            eval.metric = parameterList[["eval.metric"]],
                            objective = parameterList[["objective"]],
                            max.depth = parameterList[["max_depth"]], 
                            eta = parameterList[["eta"]], 
                            gamma = parameterList[["gamma"]], 
                            subsample = parameterList[["subsample"]], 
                            colsample_bytree = parameterList[["colsample_bytree"]],
                            print_every_n = 10,
                            min_child_weight = parameterList[["min_child"]])
    
    return(xgboostModel)
    
  }, label_apply = label_train, data_apply = data_train)
  
  names(models) <- paste(params$data,params$label,sep = "_")
  
  return(models)
}

get_names <- function(df,nameList = prettyNames){
  
  # Get pretty human names from a named character vector (prettyNames by default)
  
  # Pretty straigthforward for just a character vector:
  
  if(class(df) == "character"){
    out <- ifelse(df %in% names(nameList),
                  nameList[match(df, names(nameList))],
                  df)
    return(out)
  }
  
  # A bit more convoluted for a dataframe:
  
  out <- df %>%
    mutate_if(is.character,list(function(x,nl = nameList){
      out <- ifelse(x %in% names(nl),
                    nl[match(x, names(nl))],
                    x)
      return(out)
    }))%>%
    mutate_if(is.factor,list(function(x,nl = nameList){
      y <- as.character(x)
      
      out <- factor(ifelse(y %in% names(nl),
                    nl[match(y, names(nl))],
                    y),
                    levels = unique(nl[match(levels(x), names(nl))]))
      return(out)
    }))
  
  # Also check column names
  
  if(!is.null(names(out))){
  names(out) <- ifelse(names(out) %in% names(nameList),
                          nameList[match(colnames(out), names(nameList))],
                          names(out))
  }
  
  # ... And row names
  if(!is.null(rownames(out))){
    rownames(out) <- ifelse(rownames(out) %in% nameList,
                            nameList[match(rownames(out), names(nameList))],
                            rownames(out))
  }
  
  return(out)
}

get_exclusions <- function(df, ci = F){
  
  # Return the number of excluded and remaining records after the application of each exclusion criteria
  
  out <- data.frame("N_meeting_criteria"= colSums(dplyr::select(df,
                                                         starts_with("incl_"))),
                    "N_excluded" = as.numeric(table(df$inclGroup)),
                    "pct_excl" = NA) %>%
    mutate(var = rownames(.),
           "N_remain" = nrow(df) - cumsum(N_excluded)) %>%
    dplyr::select(var, everything(),-N_meeting_criteria)
  
  # Get bootstrapped CIs for the percentage excluded if desired
  if(ci){
    pcts <- apply(out,1,function(x){
      paste_boot_ci(c(rep(1,as.numeric(x["N_excluded"])),
                      rep(0,as.numeric(x["N_remain"]))),newline = F)
    })
  }else{
   pcts <- round(as.numeric(out$N_excluded)/as.numeric(out$N_remain)*100,1)
  }

  
  out$pct_excl <- pcts
  
  return(out)
}

plot_calibration <- function(p,l){
  
  # Plot overall calibration curves using the rms package
  
  dat <- data.frame(pred = p, lab = as.factor(l)) 
  
  lrm <- lrm(lab ~ pred,data = dat, x=T,y=T)
  
  return(plot(calibrate(lrm),
              subtitles=T,
              legend = F))
}

val_prob <- function(p,l, g = NULL,m = 0){
  
  # Use a different function from the rms package to get calibration curves stratified by some grouping vairable
  
  dat <- data.frame(pred = p, lab = l) 
  
  mod <- glm(lab ~ pred, data = dat, family = binomial)
  
  out <- val.prob(fitted(mod),l, logistic.cal = F, group = g, nmin = m)
  
  return(out)
}

## Modified version of the plot method contained in RMS with a 
# cosmetic change to reduce the size of the line of ideal calibration,
# and to set flexible plot limits for cases where outcomes are rare
# This is mostly Frank Harrels code if you're wondering why it's so comparatively elegant.
val_prob_plot <- function(x, 
                   xlab="Predicted Probability", 
                   ylab="Actual Probability",
                   lim=c(0,1), statloc=lim, stats=1:12, cex=.5, 
                   lwd.overall=4, quantiles=c(0.05,0.95),
                   flag=function(stats) ifelse(
                     stats[,'ChiSq2'] > qchisq(.99,2) |
                       stats[,'B ChiSq'] > qchisq(.99,1),'*',' '), ...)
{
  stats <- x$stats[,stats,drop=FALSE]
  lwd <- rep(par('lwd'), nrow(stats))
  lwd[dimnames(stats)[[1]]=='Overall'] <- lwd.overall
  curves <- x$cal.curves
  
  # Add some code here to calculate appropriate plot limits
  d <- unlist(curves)
  pad = 0.1
  xl <- c(max(min(d[grepl(".x",names(d))])-pad,0),
         min(max(d[grepl(".x",names(d))])+pad,1))
  yl <- c(max(min(d[grepl(".y",names(d))])-pad,0),
         min(max(d[grepl(".y",names(d))])+pad,1))
  
  labcurve(curves, pl=TRUE, xlim=xl, ylim=yl, 
           xlab=xlab, ylab=ylab, cex=cex, lwd=lwd, ...)
  abline(a=0, b=1, lwd=1, col=gray(.86)) ## Changed lwd to 1 from 6
  if(is.logical(statloc) && ! statloc) return(invisible())
  
  if(length(quantiles))
  {
    limits <- x$quantiles
    quant <- round(as.numeric(dimnames(limits)[[2]]),3)
    w <- quant %in% round(quantiles,3)
    if(any(w)) for(j in 1:nrow(limits))
    {
      qu <- limits[j,w]
      scat1d(qu, y=approx(curves[[j]], xout=qu, ties=mean)$y)
    }
  }
  
  xx <- statloc[1]; y <- statloc[2]
  for(i in 0:ncol(stats))
  {
    column.text <- if(i==0) c('Group',
                              paste(flag(stats),dimnames(stats)[[1]],sep='')) else
                                c(dimnames(stats)[[2]][i], 
                                  format(round(stats[,i], if(i %in% c(4:5,11))1 else 3)))
    cat(column.text, '\n')
    text(xx, y, paste(column.text, collapse='\n'), adj=0, cex=cex)
    xx <- xx + (1 + .8 * max(nchar(column.text))) * cex * par('cxy')[1]
  }
  invisible()
}

clean_pins <- function(d, split = "\\|"){
  
  #Do some basic cleanup and validation to exclude incorrectly formatted personal identification numbers
  
  # if mutliple PINs, generate a vector
  t <- strsplit(d,split)
  
  t <- sapply(t,function(x){
    # If 10 characters long, in almost all cases the century is missing. 
    # We can luckily assume a 20th century birthday since we're excluding kiddos <18 years old anyway.
    o <- ifelse(nchar(x) == 10,paste0("19",x),x)
    o <- o[nchar(o) == 12]
    # If there multiple non-identical PINs, set the value to NA to be excluded
    o <- ifelse(length(unique(o)) == 1,o[1],NA)
    return(o)
  })
  return(t)
}

trunc_rnorm <- function(n,m,sd,l,u){
  
  # Generate a vector of random values from a normal distribution truncated to some lower and upper boundary values
  # Used for generating synthetic data
  
  qnorm(runif(n,
              pnorm(l,m,sd),
              pnorm(u,m,sd)),
        m,sd)
}

pool_vals <- function(vals,labs,boot_type = "roc"){
  
  out <- character(0)
  
  for(i in labs){
    t <- apply(vals,2,boot_auc,l = i,type = boot_type, print = F)
    
    ests <- unlist(lapply(t,function(x){
      x$t0
    }))
    
    ses <- unlist(lapply(t,function(x){
      #transform CIs to SEs -- Assuming normality here
      (x$percent[5]  - x$percent[4])/3.92
    }))
    
    #within variance
    wv <- mean(ses^2)
    
    #between variance
    bv <- sum(mean(ests)-ests)/(length(ests)-1)
    
    # Total variance
    tv <- wv + bv + bv/length(ests)
    
    sd <- sqrt(tv)
    
    ci_paste <- paste0(round(mean(ests),2),"\n(",
                       round(mean(ests)-sd*1.96,2),"-",
                       round(mean(ests)+sd*1.96,2),")")
    
    out <- c(out,ci_paste)
  }
  
  names(out) <- names(labs)
  
  return(out)
}

move_malformatted <- function(str){
  split = str_split(str,pattern = "-")
  out = ifelse(sapply(split, "[[", 1) == "100" & sapply(split,length)>= 3,
               paste(sapply(split, "[[", 2),
                     sapply(split, "[[", 1),
                     sapply(split, "[[", 3),
                     sep = "-"),
               str)
  return(out)
}

get_min_after <- function(ts,tl){
  t_split = str_split(tl,"\\|",simplify = T)
  t_split = ymd_hms(t_split)
  t_split = t_split[which(t_split >= ts-minutes(1))]
  return(min(t_split))
}