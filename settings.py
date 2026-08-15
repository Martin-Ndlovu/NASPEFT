
# task specific settings
TASK_SETTINGS = {
    "mrpc": {
        "num_epochs": 20,
        "patience": 10,
        "save_type": "epoch",
        "logging_steps": 100,
        # in low resource, we need to set this accordingly 25 for 10ep in 300; 50 for 20ep in 300.
        "num_steps_per_save": 115,
        # 20 for 20epocch in 100. 160 for 20 epoch in 1000. 80for 20 epoch in 500.20 for 20 epochs in 100
        # <- set this to the nadir point by doing some random sampling in the task. Not required for single task optimization
        "ref_point": [26., 0.7],
        "objectives": ["param", "acc"],
        # "low_resource": 100
    },
    "sst2": {
        "num_epochs": 5, # 20 for bert and roberta. 5 for t5.
        "patience": 15,
        "save_type": "epoch",
        "logging_steps": 100,
        # in low resource, we need to set this accordingly 25 for 10ep in 300; 50 for 20ep in 300.
        "num_steps_per_save": 1,
        # 20 for 20epocch in 100. 160 for 20 epoch in 1000. 80for 20 epoch in 500.20 for 20 epochs in 100
        # <- set this to the nadir point by doing some random sampling in the task. Not required for single task optimization
        "ref_point": [26., 0.7],
        "objectives": ["param", "acc"]
    },
    "qnli": {
        "num_epochs": 4,
        "patience": 10,
        "save_type": "epoch",
        "num_steps_per_save": 1,
        "logging_steps": 100,
        # in low resource, we need to set this accordingly 25 for 10ep in 300; 50 for 20ep in 300.
        # 20 for 20epocch in 100. 160 for 20 epoch in 1000. 80for 20 epoch in 500.20 for 20 epochs in 100
        "ref_point": [26., 0.7],
        # <- set this to the nadir point by doing some random sampling in the task. Not required for single task optimization
        "objectives": ["param", "acc"]
    },
    "qqp": {
        "num_epochs": 2,
        "patience": 10,
        "save_type": "epoch",
        "num_steps_per_save": 1000,
        "logging_steps": 100,
        # in low resource, we need to set this accordingly 25 for 10ep in 300; 50 for 20ep in 300.
        # 20 for 20epocch in 100. 160 for 20 epoch in 1000. 80for 20 epoch in 500.20 for 20 epochs in 100
        "ref_point": [26., 0.7],
        # <- set this to the nadir point by doing some random sampling in the task. Not required for single task optimization
        "objectives": ["param", "acc"],
        # "low_resource": 100
    },
    "mnli": {
        "num_epochs": 3,
        "patience": 10,
        "save_type": "epoch",
        "num_steps_per_save": 1,
        "logging_steps": 100,
        # in low resource, we need to set this accordingly 25 for 10ep in 300; 50 for 20ep in 300.
        # 20 for 20epocch in 100. 160 for 20 epoch in 1000. 80for 20 epoch in 500.20 for 20 epochs in 100
        "ref_point": [26., 0.7],
        # <- set this to the nadir point by doing some random sampling in the task. Not required for single task optimization
        "objectives": ["param", "acc"]
    },
    "cola": {
        "num_epochs": 15,
        "patience": 10,
        "save_type": "epoch",
        "num_steps_per_save": 1,
        "logging_steps": 100,
        # in low resource, we need to set this accordingly 25 for 10ep in 300; 50 for 20ep in 300.
        # 20 for 20epocch in 100. 160 for 20 epoch in 1000. 80for 20 epoch in 500.20 for 20 epochs in 100
        "ref_point": [26., 0.7],
        # <- set this to the nadir point by doing some random sampling in the task. Not required for single task optimization
        "objectives": ["param", "acc"]
    },
    "rte": {
        "num_epochs": 40,
        "patience": 10,
        "save_type": "epoch",
        "num_steps_per_save": 1,
        "logging_steps": 100,
        # in low resource, we need to set this accordingly 25 for 10ep in 300; 50 for 20ep in 300.
        # 20 for 20epocch in 100. 160 for 20 epoch in 1000. 80for 20 epoch in 500.20 for 20 epochs in 100
        "ref_point": [26., 0.7],
        # <- set this to the nadir point by doing some random sampling in the task. Not required for single task optimization
        "objectives": ["param", "acc"]
    },
    "stsb": {
        "num_epochs": 40,
        "patience": 10,
        "save_type": "epoch",
        "num_steps_per_save": 1,
        "logging_steps": 100,
        # in low resource, we need to set this accordingly 25 for 10ep in 300; 50 for 20ep in 300.
        # 20 for 20epocch in 100. 160 for 20 epoch in 1000. 80for 20 epoch in 500.20 for 20 epochs in 100
        "ref_point": [26., 0.7],
        # <- set this to the nadir point by doing some random sampling in the task. Not required for single task optimization
        "objectives": ["param", "acc"]
    },
    
    "wikitext": {
        "num_epochs": 3,
        "patience": 3,
        "save_type": "steps",
        "logging_steps": 100,
        "num_steps_per_save":1000,
        "ref_point": [10.0, 100.0],  # [param %, perplexity]
        "objectives": ["param", "perplexity"],
        "low_resource": None,
    }

    # Add new task configs here
}

TASK_SETTINGS_SO = {
    "mrpc": {
        "num_epochs": 20,
        "patience": 10,
        "save_type": "epoch",
        # in low resource, we need to set this accordingly 25 for 10ep in 300; 50 for 20ep in 300.
        "num_steps_per_save": 115,
        # 20 for 20epocch in 100. 160 for 20 epoch in 1000. 80for 20 epoch in 500.20 for 20 epochs in 100
        # <- set this to the nadir point by doing some random sampling in the task. Not required for single task optimization
        "ref_point": [0.7],
        "objectives": ["acc"],
        # "low_resource": 100
    },
    "sst2": {
        "num_epochs": 5, # 20 for bert and roberta. 5 for t5.
        "patience": 10,
        "save_type": "epoch",
        # in low resource, we need to set this accordingly 25 for 10ep in 300; 50 for 20ep in 300.
        "num_steps_per_save": 1,
        # 20 for 20epocch in 100. 160 for 20 epoch in 1000. 80for 20 epoch in 500.20 for 20 epochs in 100
        # <- set this to the nadir point by doing some random sampling in the task. Not required for single task optimization
        "ref_point": [0.7],
        "objectives": ["acc"]
    },
    "qnli": {
        "num_epochs": 20,
        "patience": 10,
        "save_type": "epoch",
        "num_steps_per_save": 1,
        # in low resource, we need to set this accordingly 25 for 10ep in 300; 50 for 20ep in 300.
        # 20 for 20epocch in 100. 160 for 20 epoch in 1000. 80for 20 epoch in 500.20 for 20 epochs in 100
        "ref_point": [0.7],
        # <- set this to the nadir point by doing some random sampling in the task. Not required for single task optimization
        "objectives": ["acc"]
    },
    "qqp": {
        "num_epochs": 20,
        "patience": 10,
        "save_type": "epoch",
        "num_steps_per_save": 115,
        # in low resource, we need to set this accordingly 25 for 10ep in 300; 50 for 20ep in 300.
        # 20 for 20epocch in 100. 160 for 20 epoch in 1000. 80for 20 epoch in 500.20 for 20 epochs in 100
        "ref_point": [0.7],
        # <- set this to the nadir point by doing some random sampling in the task. Not required for single task optimization
        "objectives": ["acc"],
        # "low_resource": 100
    },
    "mnli": {
        "num_epochs": 3,
        "patience": 10,
        "save_type": "epoch",
        "num_steps_per_save": 1,
        # in low resource, we need to set this accordingly 25 for 10ep in 300; 50 for 20ep in 300.
        # 20 for 20epocch in 100. 160 for 20 epoch in 1000. 80for 20 epoch in 500.20 for 20 epochs in 100
        "ref_point": [0.7],
        # <- set this to the nadir point by doing some random sampling in the task. Not required for single task optimization
        "objectives": ["acc"]
    },
    "cola": {
        "num_epochs": 20,
        "patience": 10,
        "save_type": "epoch",
        "num_steps_per_save": 1,
        # in low resource, we need to set this accordingly 25 for 10ep in 300; 50 for 20ep in 300.
        # 20 for 20epocch in 100. 160 for 20 epoch in 1000. 80for 20 epoch in 500.20 for 20 epochs in 100
        "ref_point": [0.7],
        # <- set this to the nadir point by doing some random sampling in the task. Not required for single task optimization
        "objectives": ["acc"]
    },
    "rte": {
        "num_epochs": 40,
        "patience": 10,
        "save_type": "epoch",
        "num_steps_per_save": 1,
        # in low resource, we need to set this accordingly 25 for 10ep in 300; 50 for 20ep in 300.
        # 20 for 20epocch in 100. 160 for 20 epoch in 1000. 80for 20 epoch in 500.20 for 20 epochs in 100
        "ref_point": [0.7],
        # <- set this to the nadir point by doing some random sampling in the task. Not required for single task optimization
        "objectives": ["acc"]
    },
    "stsb": {
        "num_epochs": 20,
        "patience": 10,
        "save_type": "epoch",
        "num_steps_per_save": 1,
        # in low resource, we need to set this accordingly 25 for 10ep in 300; 50 for 20ep in 300.
        # 20 for 20epocch in 100. 160 for 20 epoch in 1000. 80for 20 epoch in 500.20 for 20 epochs in 100
        "ref_point": [0.7],
        # <- set this to the nadir point by doing some random sampling in the task. Not required for single task optimization
        "objectives": ["acc"]
    },
    "wikitext": {
        "num_epochs": 3,
        "patience": 3,
        "save_type": "steps",
        "logging_steps": 1,
        "num_steps_per_save": 5,
        "ref_point": [100.0],  # [perplexity]
        "objectives": ["perplexity"],
        "low_resource": None,
    }
    # Add new task configs here
}