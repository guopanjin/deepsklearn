
#12157895
train_data_path="~/.deepsklearn/data/aliexpress/AliExpress_NL/train.csv"

#5559302
test_data_path="~/.deepsklearn/data/aliexpress/AliExpress_NL/test.csv"



categorical_feature_columns={'categorical_1': 5,
                             'categorical_2': 20,
                             'categorical_3': 35,
                             'categorical_4': 10,
                             'categorical_5': 10,
                             'categorical_6': 35,
                             'categorical_7': 185,
                             'categorical_8': 40,
                             'categorical_9': 40,
                             'categorical_10': 10,
                             'categorical_11': 10,
                             'categorical_12': 10,
                             'categorical_13': 10,
                             'categorical_14': 10,
                             'categorical_15': 10,
                             'categorical_16': 10}
numerical_feature_columns=['numerical_1', 'numerical_2', 'numerical_3', 'numerical_4', 'numerical_5', 'numerical_6', 'numerical_7', 'numerical_8', 'numerical_9', 'numerical_10', 'numerical_11', 'numerical_12', 'numerical_13', 'numerical_14', 'numerical_15', 'numerical_16', 'numerical_17', 'numerical_18', 'numerical_19', 'numerical_20', 'numerical_21', 'numerical_22', 'numerical_23', 'numerical_24', 'numerical_25', 'numerical_26', 'numerical_27', 'numerical_28', 'numerical_29', 'numerical_30', 'numerical_31', 'numerical_32', 'numerical_33', 'numerical_34', 'numerical_35', 'numerical_36', 'numerical_37', 'numerical_38', 'numerical_39', 'numerical_40', 'numerical_41', 'numerical_42', 'numerical_43', 'numerical_44', 'numerical_45', 'numerical_46', 'numerical_47', 'numerical_48', 'numerical_49', 'numerical_50', 'numerical_51', 'numerical_52', 'numerical_53', 'numerical_54', 'numerical_55', 'numerical_56', 'numerical_57', 'numerical_58', 'numerical_59', 'numerical_60', 'numerical_61', 'numerical_62', 'numerical_63']

label_columns=["click","conversion"]



###train data sample
'''
search_id,categorical_1,categorical_2,categorical_3,categorical_4,categorical_5,categorical_6,categorical_7,categorical_8,categorical_9,categorical_10,categorical_11,categorical_12,categorical_13,categorical_14,categorical_15,categorical_16,numerical_1,numerical_2,numerical_3,numerical_4,numerical_5,numerical_6,numerical_7,numerical_8,numerical_9,numerical_10,numerical_11,numerical_12,numerical_13,numerical_14,numerical_15,numerical_16,numerical_17,numerical_18,numerical_19,numerical_20,numerical_21,numerical_22,numerical_23,numerical_24,numerical_25,numerical_26,numerical_27,numerical_28,numerical_29,numerical_30,numerical_31,numerical_32,numerical_33,numerical_34,numerical_35,numerical_36,numerical_37,numerical_38,numerical_39,numerical_40,numerical_41,numerical_42,numerical_43,numerical_44,numerical_45,numerical_46,numerical_47,numerical_48,numerical_49,numerical_50,numerical_51,numerical_52,numerical_53,numerical_54,numerical_55,numerical_56,numerical_57,numerical_58,numerical_59,numerical_60,numerical_61,numerical_62,numerical_63,click,conversion
3,8,0,0,0,8,4,31,0,3,0,0,1,0,0,1,0,1.0,0.0,0.10526300000000001,0.555556,1.0,0.22222199999999998,0.333333,0.22222199999999998,0.1381,0.21635,0.08975,0.18785,0.07715,0.0714,0.05165,0.05485,0.05065,0.04725,0.6195,0.73679,0.82916,0.18596,0.18492999999999998,0.07152,0.0,0.0,0.03848,0.6,0.34,0.0005099999999999999,0.779758,0.001556,0.05784299999999999,0.59633,0.442353,0.037859,0.022775,0.267322,0.296752,0.550516,0.144544,0.0,0.0,1.0,0.0,0.0,0.275016,0.1818,0.052289999999999996,0.30713,0.30307,0.249969,0.28226999999999997,0.1603,0.145979,0.8178890000000001,0.119395,0,0,0,0.47871,0.319446,0.373299,0,0
3,8,0,0,0,8,4,31,0,3,0,0,1,0,0,1,0,1.0,0.0,0.10526300000000001,0.555556,1.0,0.22222199999999998,0.333333,0.22222199999999998,0.1381,0.21635,0.08975,0.18785,0.07715,0.0714,0.05165,0.05485,0.05065,0.04725,0.6195,0.73679,0.82916,0.18596,0.18492999999999998,0.076472,0.0,0.22941599999999998,0.037857,0.64,0.0,0.02452,0.0,0.0030800000000000003,0.0,0.390989,0.334949,0.039064999999999996,0.0,0.22160100000000002,0.310253,0.584303,0.247522,0.0,0.0,1.0,0.0,0.0,0.0,0.19929000000000002,0.09746,0.34726999999999997,0.36678299999999997,0.26820900000000003,0.316278,0.1603,0.206477,0.818441,0.168989,0,0,0,0.0,0.336834,0.451851,0,0

'''

##test data sample
'''
search_id,categorical_1,categorical_2,categorical_3,categorical_4,categorical_5,categorical_6,categorical_7,categorical_8,categorical_9,categorical_10,categorical_11,categorical_12,categorical_13,categorical_14,categorical_15,categorical_16,numerical_1,numerical_2,numerical_3,numerical_4,numerical_5,numerical_6,numerical_7,numerical_8,numerical_9,numerical_10,numerical_11,numerical_12,numerical_13,numerical_14,numerical_15,numerical_16,numerical_17,numerical_18,numerical_19,numerical_20,numerical_21,numerical_22,numerical_23,numerical_24,numerical_25,numerical_26,numerical_27,numerical_28,numerical_29,numerical_30,numerical_31,numerical_32,numerical_33,numerical_34,numerical_35,numerical_36,numerical_37,numerical_38,numerical_39,numerical_40,numerical_41,numerical_42,numerical_43,numerical_44,numerical_45,numerical_46,numerical_47,numerical_48,numerical_49,numerical_50,numerical_51,numerical_52,numerical_53,numerical_54,numerical_55,numerical_56,numerical_57,numerical_58,numerical_59,numerical_60,numerical_61,numerical_62,numerical_63,click,conversion
3,8,0,0,0,8,3,25,0,0,1,0,0,0,0,1,0,1.0,0.11111099999999999,0.0,0.666667,0.666667,1.0,0.333333,0.0,0.26645,0.134375,0.11005,0.100925,0.055875,0.0497,0.04545,0.05725,0.06175,0.103225,0.4669,0.30151,0.0,0.12072999999999999,0.0,0.29488400000000003,0.21345,0.502273,0.0,0.51,0.43,0.22082,0.572446,0.056521,0.02783,0.619324,0.59353,0.085526,0.022908,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.591622,0.0,0.0,0.0,0.722715,0.668775,0.788233,0.2079,0.331495,0.324986,0.107731,0,0,0,0.17614000000000002,0.0,0.0,0,0
3,8,0,0,0,8,3,25,0,0,0,0,0,0,0,1,0,1.0,0.11111099999999999,0.0,0.666667,0.666667,1.0,0.333333,0.0,0.26645,0.134375,0.11005,0.100925,0.055875,0.0497,0.04545,0.05725,0.06175,0.103225,0.4669,0.30151,0.0,0.12072999999999999,0.0,0.069881,0.0,0.171173,0.0466,0.69,0.32,0.15294000000000002,0.7143689999999999,0.000321,0.0,0.661611,0.6372680000000001,0.033982,0.0,0.12309400000000001,0.825714,0.825714,0.0,0.0,0.0,0.0,0.0,0.0,0.370384,0.12666,0.05435,0.14715999999999999,0.171665,0.154851,0.061536,0.4268,0.12755899999999998,0.613734,0.078287,0,0,0,0.53671,0.0,0.0,0,0
'''
