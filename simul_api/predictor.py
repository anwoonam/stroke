# Create your predictor here.

# from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import pandas as pd
import pickle

class MLModel:

    g_pred_path = './model/best_g_pred(LGBMR)(24v).pkl'
    se_pred_path = './model/best_se_pred(LGBMR)(24v).pkl'
    bmi_pred_path = './model/best_bmi_pred(KNNR)(24v).pkl'
    oc_pred_path = './model/best_oc_pred(KNNC)(24v).pkl'

    iv_cate_dic = {
        'Aquatic training': [
            'Conventional rehabilitation exercises + aquatic walking + NDT',
            'Underwater walking',
            'Underwater walking + Conventional rehabilitation exercises'
        ],
        'Combine aerobic and breathing training': [
            'Conventional rehabilitation exercises + Cycling ergometer + RIMT + REMT',
            'Joint mobilization + Cycling ergometer + ES',
            'Overground walking + PLB + Diaphragmatic breathing exercise + Conventional rehabilitation exercises',
            'RIMT + REMT + Game-based TST'
        ],
        'Combine aerobic and resistance training': [
            'High speed RT + combined aerobic and resistance training',
            'Low speed RT + combined aerobic and resistance training',
            'RIMT + Conventional rehabilitation exercises + combined aerobic and resistance training',
            'RT + Overground walking',
            'Combined aerobic and resistance training',
            'Combined aerobic and resistance training, Game-based exersice'
        ],
        'Combine inspiratory and expiratory training': [
            'Breathing exercise, Game-based exersice',
            'Conventional rehabilitation exercises + Air stacking exercise + RIMT + REMT',
            'Conventional rehabilitation exercises + NST + Diaphragmatic breathing exercise + PLB',
            'Conventional rehabilitation exercises + RIMT + Abdominal draw-in training',
            'Conventional rehabilitation exercises + RIMT + REMT',
            'Diaphragmatic breathing exercise + PLB + Conventional rehabilitation exercises',
            'Dysphagia Therapy, Breathing exercise',
            'Posture exercises + Thoracic expansion exercises + Endurance exercises + Postural drainage + Diaphragmatic breathing exercise + SegMental breathing + Long maximal inspiration + PLB + Respiratory control + RIMT + REMT + Conventional rehabilitation exercises',
            'RIMT + Conventional rehabilitation exercises + Abdominal strengthening exercise',
            'RIMT + REMT + Conventional rehabilitation exercises'
        ],
        'Ground-based aerobic training': [
            'Cycling ergometer',
            'Cycling ergometer + ES',
            'High intensity Treadmill',
            'High intensity Treadmill combined with obstacle-crossing',
            'Low intensity Treadmill combined with obstacle-crossing',
            'Low intensity treadmill',
            'Moderate intensity Treadmill',
            'Recumbent stepper',
            'Treadmill',
            'Treadmill + Conventional rehabilitation exercises'
        ],
        'Inspiratory training': [
            'Conventional rehabilitation exercises + RIMT',
            'Diaphragmatic breathing exercise + RIMT + NDT',
            'High intensity RIMT',
            'Joint mobilization + Diaphragmatic breathing exercise + NDT',
            'RIMT',
            'RIMT + Chest expansion exercise',
            'RIMT + Conventional rehabilitation exercises',
            'RIMT + ES',
            'RIMT + NDT',
            'RIMT_fixed',
            'RIMT_progressive'
        ],
        'Resistance training': [
            'Lower extremity RT',
            'RT',
            'Resistance exercise + Conventional rehabilitation exercises'
        ],
        'Robot-assisted training': [
            'Robot-assisted gait training (Feed back) + Treadmill + Conventional rehabilitation exercises',
            'Robot-assisted gait training + Conventional rehabilitation exercises + NDT',
            'Robot-assisted gait training + Treadmill + Conventional rehabilitation exercises',
            'Robot-assisted gait training + Treadmill + Overground walking'
        ]
    }
    oc_cate_dic = {
        'Activity ability': [
            'ADL',
            'Activity',
            'Fatigue',
            'QOL',
            'Sedentary',
            'TIS'
        ],
        'Balance ability': [
            'ABC',
            'BBS',
            'COP',
            'FRT',
            'TUG'
        ],
        'Cardiovascular function': [
            'Blood pressure',
            'Blood vessel',
            'Exercise tolerance',
            'HR',
            'MET',
            'Max HR',
            'Oxygen saturation',
            'Oxygen uptake',
            'Peak HR',
            'Power output',
            'RPE',
            'Resting HR',
            'Resting SBP'
        ],
        'Echocardiography': [
            'LVEED',
            'LVEF',
            'LVESD',
            'Lateral mitral annulus e′',
            'Lateral tricuspid annulus e′',
            'Right atrial emptying fraction',
            'Transmitral inflow A',
            'Transmitral inflow E'
        ],
        'Electrocardiogram': [
            'P Wave duration',
            'PR interval',
            'QRS interval',
            'Qt duration',
            'Qtc duration',
            'Tp-e interval',
            'Tp-e/QT ratio',
            'Tp-e/QTc ratio'
        ],
        'Gait function': [
            '10MWT',
            '6MWT',
            'Cadence',
            'Gait cycle',
            'Gait energy',
            'Gait index',
            'Gait speed',
            'Length of walking distance',
            'Respiratory exercise capacity',
            'Spatial step symmetry ratio',
            'Stance_affected',
            'Stance_non affected',
            'Step Length_affected',
            'Step Length_non affected',
            'Step test',
            'Step Time_affected',
            'Step Time_non affected',
            'Step per day', 'Stride length',
            'Swing_affected',
            'Swing_non affected',
            'Symmetry Ratio',
            'Temporal symmetry ratio',
            'Walk speed',
            'Work load'
        ],
        'Mass': [
            'BMI',
            'Body Mass',
            'Fat free mass',
            'Skeletal muscle mass',
            'Total body lean mass',
            'Total body percentage fat'
        ],
        'Mental': [
            'Cognitive',
            'Depression'
        ],
        'Motor function': [
            'Craniovertebral angle',
            'Dysphagia',
            'FMA',
            'Oral intake',
            'Swallowing ability',
            'Trunk'
        ],
        'Muscular endurance': [
            'Test duration'
        ],
        'Muscular strength': [
            'Diaphragm',
            'External intercostal muscle',
            'Hand grip',
            'Intensity',
            'Lower extremity',
            'Sit-to-stand seconds',
            'Stair climb seconds',
            'Upper extremity'
        ],
        'Respiratory function': [
            'Chest expansion',
            'FEF25–75',
            'FEV1',
            'FEV1/FVC',
            'FVC',
            'MEP',
            'MIP',
            'MMEF',
            'MVV',
            'PCF',
            'PEF',
            'PIF',
            'Penetration-aspiration',
            'Respiratory endurance',
            'SVC',
            'TV',
            'VC'
        ]
    }

    iv_sub_cate = [
        'Breathing exercise, Game-based exersice',
        'Combined aerobic and resistance training',
        'Combined aerobic and resistance training, Game-based exersice',
        'Conventional rehabilitation exercises + Air stacking exercise + RIMT + REMT',
        'Conventional rehabilitation exercises + Cycling ergometer + RIMT + REMT',
        'Conventional rehabilitation exercises + NST + Diaphragmatic breathing exercise + PLB',
        'Conventional rehabilitation exercises + RIMT',
        'Conventional rehabilitation exercises + RIMT + Abdominal draw-in training',
        'Conventional rehabilitation exercises + RIMT + REMT',
        'Conventional rehabilitation exercises + aquatic walking + NDT',
        'Cycling ergometer',
        'Cycling ergometer + ES',
        'Diaphragmatic breathing exercise + PLB + Conventional rehabilitation exercises',
        'Diaphragmatic breathing exercise + RIMT + NDT',
        'Dysphagia Therapy, Breathing exercise',
        'High intensity RIMT',
        'High intensity Treadmill',
        'High intensity Treadmill combined with obstacle-crossing',
        'High speed RT + combined aerobic and resistance training',
        'Joint mobilization + Cycling ergometer + ES',
        'Joint mobilization + Diaphragmatic breathing exercise + NDT',
        'Low intensity Treadmill combined with obstacle-crossing',
        'Low intensity treadmill',
        'Low speed RT + combined aerobic and resistance training',
        'Lower extremity RT',
        'Moderate intensity Treadmill',
        'Overground walking + PLB + Diaphragmatic breathing exercise + Conventional rehabilitation exercises',
        'Posture exercises + Thoracic expansion exercises + Endurance exercises + Postural drainage + Diaphragmatic breathing exercise + SegMental breathing + Long maximal inspiration + PLB + Respiratory control + RIMT + REMT + Conventional rehabilitation exercises',
        'RIMT',
        'RIMT + Chest expansion exercise',
        'RIMT + Conventional rehabilitation exercises',
        'RIMT + Conventional rehabilitation exercises + Abdominal strengthening exercise',
        'RIMT + Conventional rehabilitation exercises + combined aerobic and resistance training',
        'RIMT + ES',
        'RIMT + NDT',
        'RIMT + REMT + Conventional rehabilitation exercises',
        'RIMT + REMT + Game-based TST',
        'RIMT_fixed',
        'RIMT_progressive',
        'RT',
        'RT + Overground walking',
        'Recumbent stepper',
        'Resistance exercise + Conventional rehabilitation exercises',
        'Robot-assisted gait training (Feed back) + Treadmill + Conventional rehabilitation exercises',
        'Robot-assisted gait training + Conventional rehabilitation exercises + NDT',
        'Robot-assisted gait training + Treadmill + Conventional rehabilitation exercises',
        'Robot-assisted gait training + Treadmill + Overground walking',
        'Treadmill',
        'Treadmill + Conventional rehabilitation exercises',
        'Underwater walking',
        'Underwater walking + Conventional rehabilitation exercises'
    ]
    oc_sub_cate = [
        '10MWT',
        '6MWT',
        'ABC',
        'ADL',
        'Activity',
        'BBS',
        'BMI',
        'Blood pressure',
        'Blood vessel',
        'Body Mass',
        'COP',
        'Cadence',
        'Chest expansion',
        'Cognitive',
        'Craniovertebral angle',
        'Depression',
        'Diaphragm',
        'Dysphagia',
        'Exercise tolerance',
        'External intercostal muscle',
        'FEF25–75',
        'FEV1',
        'FEV1/FVC',
        'FMA',
        'FRT',
        'FVC',
        'Fat free mass',
        'Fatigue',
        'Gait cycle',
        'Gait energy',
        'Gait index',
        'Gait speed',
        'HR',
        'Hand grip',
        'Intensity',
        'LVEED',
        'LVEF',
        'LVESD',
        'Lateral mitral annulus e′',
        'Lateral tricuspid annulus e′',
        'Length of walking distance',
        'Lower extremity',
        'MEP',
        'MET',
        'MIP',
        'MMEF',
        'MVV',
        'Max HR',
        'Oral intake',
        'Oxygen saturation',
        'Oxygen uptake',
        'P Wave duration',
        'PCF',
        'PEF',
        'PIF',
        'PR interval',
        'Peak HR',
        'Penetration-aspiration',
        'Power output',
        'QOL',
        'QRS interval',
        'Qt duration',
        'Qtc duration',
        'RPE',
        'Respiratory endurance',
        'Respiratory exercise capacity',
        'Resting HR',
        'Resting SBP',
        'Right atrial emptying fraction',
        'SVC',
        'Sedentary',
        'Sit-to-stand seconds',
        'Skeletal muscle mass',
        'Spatial step symmetry ratio',
        'Stair climb seconds',
        'Stance_affected',
        'Stance_non affected',
        'Step Length_affected',
        'Step Length_non affected',
        'Step Time_affected',
        'Step Time_non affected',
        'Step per day',
        'Step test',
        'Stride Length',
        'Swallowing ability',
        'Swing_affected',
        'Swing_non affected',
        'Symmetry Ratio',
        'TIS',
        'TUG',
        'TV',
        'Temporal symmetry ratio',
        'Test duration',
        'Total body lean mass',
        'Total body percentage fat',
        'Tp-e interval',
        'Tp-e/QT ratio',
        'Tp-e/QTc ratio',
        'Transmitral inflow A',
        'Transmitral inflow E',
        'Trunk',
        'Upper extremity',
        'VC',
        'Walk speed',
        'Work load'
    ]

    @staticmethod
    def load_model(model_path):
        with open(model_path, 'rb') as file:
            return pickle.load(file)

    @staticmethod
    def oc_predict(age, bmi, intervention_2, session_n, training_p, training_t, gender):
        # 머신러닝 모델에 입력 데이터 전달하여 예측
        # 예측 결과 반환

        iv_sub_cate = MLModel.iv_sub_cate
        oc_sub_cate = MLModel.oc_sub_cate
        oc_cate_dic = MLModel.oc_cate_dic

        iv_sub_num = iv_sub_cate.index(intervention_2)
        age_c = age//10 + 5

        bmi_c_pred_data = pd.DataFrame({'Gender': [gender], 'Age': [age_c] })
        bmi_c = MLModel.load_model(MLModel.bmi_pred_path).predict(bmi_c_pred_data)

        # Creating a DataFrame with the input data
        input_data = pd.DataFrame({'intervention_2': [iv_sub_num], 'Gender': [gender], 'Age': [age], 'BMI': [bmi], 'Age_C': [age_c], 'BMI_C': [bmi_c], 'Session_N': [session_n], 'Training_P': [training_p], 'Training_T': [training_t] })

        # Load the trained model and predict
        g = MLModel.load_model(MLModel.g_pred_path).predict(input_data)
        se = MLModel.load_model(MLModel.se_pred_path).predict(input_data)

        # Creating a DataFrame with the input data
        g_se_data = pd.DataFrame({'g': [g], 'se': [se], 'intervention_2': [iv_sub_num], 'Gender': [gender], 'Age': [age], 'BMI': [bmi], 'Age_C': [age_c], 'BMI_C': [bmi_c], 'Session_N': [session_n], 'Training_P': [training_p], 'Training_T': [training_t]})

        # Load the models
        models = {
            'decision_tree': MLModel.load_model(MLModel.oc_pred_path),
        }

        # Loop through each model and make predictions
        for _, model in models.items():
            pred_oc_sub_num = model.predict(g_se_data)
        
        pred_oc_sub = oc_sub_cate[int(pred_oc_sub_num[0])]
        pred_oc_main = None
        for m, s in oc_cate_dic.items():
            if pred_oc_sub in s:
                pred_oc_main = m
                break

        return [pred_oc_main, pred_oc_sub]
    
    @staticmethod
    def iv_predict(age, bmi, outcome_2, session_n, training_p, training_t, gender):
        # 머신러닝 모델에 입력 데이터 전달하여 예측
        # 예측 결과 반환

        iv_sub_cate = MLModel.iv_sub_cate
        oc_sub_cate = MLModel.oc_sub_cate
        iv_cate_dic = MLModel.iv_cate_dic

        age_c = age//10 + 5

        bmi_c_pred_data = pd.DataFrame({'Gender': [gender], 'Age': [age_c] })
        bmi_c = MLModel.load_model(MLModel.bmi_pred_path).predict(bmi_c_pred_data)
            
        # Function to get user input and predict the outcome
        # Load the trained model and predict
        g_model = MLModel.load_model(MLModel.g_pred_path)
        se_model = MLModel.load_model(MLModel.se_pred_path)

        # Load the models
        models = {
            'decision_tree': MLModel.load_model(MLModel.oc_pred_path),
        }

        oc_sub_num = oc_sub_cate.index(outcome_2)

        # Loop through each model and make predictions
        for _, model in models.items():
            correct_outcome = False    
            iv_sub_num = 0
            while not correct_outcome and iv_sub_num <= 50:
                input_data = pd.DataFrame({'intervention_2': [str(iv_sub_num)], 'Gender': [gender], 'Age': [age], 'BMI': [bmi], 'Age_C': [age_c], 'BMI_C': [bmi_c], 'Session_N': [session_n], 'Training_P': [training_p], 'Training_T': [training_t] })

                g = g_model.predict(input_data)
                se = se_model.predict(input_data)

                # Creating a DataFrame with the input data
                g_se_data = pd.DataFrame({'g': [g], 'se': [se], 'intervention_2': [str(iv_sub_num)], 'Gender': [gender], 'Age': [age], 'BMI': [bmi], 'Age_C': [age_c], 'BMI_C': [bmi_c], 'Session_N': [session_n], 'Training_P': [training_p], 'Training_T': [training_t] })            
                pred_oc_sub_num = model.predict(g_se_data)

                if int(pred_oc_sub_num) == oc_sub_num:
                    correct_outcome = True
                    break
                else:
                    iv_sub_num += 1

        pred_iv_sub = iv_sub_cate[iv_sub_num]
        pred_iv_main = None
        for m, s in iv_cate_dic.items():
            if pred_iv_sub in s:
                pred_iv_main = m
                break

        return [pred_iv_main, pred_iv_sub]