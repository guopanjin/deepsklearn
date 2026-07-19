import json
import subprocess
import sys
import time
from pathlib import  Path
import streamlit as st
#remove top-padding
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)
with st.sidebar:
    model_name=st.selectbox("Model",
                       ["deepfm","lr","dnn","wide_deep","dcn"],
                       index=0
                       )
    train_data_path=st.text_input(label="train_data_path",value="~/.deepsklearn/data/criteo/debug_train.csv")
    validation_data_path=st.text_input(label="validation_data_path",value="~/.deepsklearn/data/criteo/debug_validation.csv")
    feature_config_path=st.text_input("feature_config_path",value="~/.deepsklearn/data/criteo/criteo_feature_column.json")
    label_config_path=st.text_input("feature_config_path",value="~/.deepsklearn/data/criteo/criteo_label_column.json")
    train_batch_size = st.number_input("train_batch_size", value=1000)
    validation_batch_size=st.number_input("validation_batch_size",value=1000)
    device = st.text_input("device", 'cpu')
    model_path=st.text_input(label="model_path",value=f"~/.deepsklearn/data/criteo/{model_name}/")

st.title("deepsklearn training dashboard")
left,center1,center2,right=st.columns([1,1,1,1])
with center1:
    start_training=st.button("start_training", type="primary")
with center2:
    stop_training=st.button("stop_training", type="primary")
st.subheader("train config")
train_config={
    "model_name":model_name,
    "train_data_path":train_data_path,
    "validation_data_path":validation_data_path,
    "feature_config":feature_config_path,
    "label_config":label_config_path,
    "train_batch_size":train_batch_size,
    "validation_batch_size":validation_batch_size,
    "model_path":model_path,
    "device":device
}
st.json(train_config)
config_path=f"../runtime_configs/{model_name}.json"
log_path = f"../logs/{model_name}.log"
if start_training:
    # same file name will overwrite
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(train_config, f, indent=2, ensure_ascii=False)
    cmd=[
        sys.executable,#current python commond
        "-u",# unbufferd mode
        "../scripts/train_non_sequence.py",
        "--config",
        config_path
    ]
    #start sub process to run the train code
    with open(log_path,"a",encoding="utf-8",buffering=1) as log_file:
        process=subprocess.Popen(cmd,
                                 stdout=log_file,
                                 stderr=log_file
                                 )
        print(f"traning process:{process}")
if stop_training:
    pass


st.subheader("train log")
if Path(log_path).exists():
    with open(log_path,"r",encoding="utf-8") as f:
        log_text=f.readlines()
        if len(log_text)<1:
            log_text="wainting for training log"
    st.code(log_text[-1000:])



