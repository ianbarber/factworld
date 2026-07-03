# Staged curriculum — architecture/batch grid sweep

grid=d_model:128;n_layers:2;batch:8;steps:60;seeds:0

## Per-config final eval (mean over seeds)

| config | bind L16 | recall easy | recall med | recall hard | comp p5 | comp p16 | p16 holder | p16 value | p16 scaffold | final loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| batch8_d_model128_n_layers2_steps60_train_n8000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 6.723 |

## Raw results

```json
[
  {
    "cfg": {
      "d_model": 128,
      "n_layers": 2,
      "batch": 8,
      "steps": 60,
      "seed": 0,
      "train_n": 8000
    },
    "label": "batch8_d_model128_n_layers2_steps60_train_n8000",
    "stage_records": [
      {
        "phase": 0,
        "weights": {
          "binding": 0.5,
          "recall_easy": 0.5
        },
        "steps": 24,
        "arm_counts": {
          "binding": 4000,
          "recall_easy": 4000
        },
        "final_loss": 6.866908550262451,
        "loss_curve": [],
        "eval": {
          "binding_L16": {
            "overall": 0.0,
            "prefix": {
              "0": 1.0,
              "1": 0.0,
              "2": 0.0
            },
            "holder_acc": 0.0,
            "value_acc": 0.0
          },
          "recall_easy_L4": {
            "overall": 0.0,
            "prefix": {
              "0": 1.0,
              "1": 0.0,
              "2": 0.0
            },
            "holder_acc": 0.0,
            "value_acc": 0.0
          },
          "recall_med_L8": {
            "overall": 0.0,
            "prefix": {
              "0": 1.0,
              "1": 0.0,
              "2": 0.0
            },
            "holder_acc": 0.0,
            "value_acc": 0.0
          },
          "recall_hard_L16": {
            "overall": 0.0,
            "prefix": {
              "0": 1.0,
              "1": 0.0,
              "2": 0.0
            },
            "holder_acc": 0.0,
            "value_acc": 0.0
          },
          "composite_p5_L16": {
            "overall": 0.0,
            "prefix": {
              "0": 1.0,
              "1": 0.0,
              "2": 0.0
            },
            "holder_acc": 0.0,
            "value_acc": 0.0
          },
          "composite_p16_L16": {
            "overall": 0.0,
            "prefix": {
              "0": 1.0,
              "1": 0.0,
              "2": 0.0
            },
            "holder_acc": 0.0,
            "value_acc": 0.0
          },
          "composite_p16_scaffolded": {
            "scaffolded_value": 0.0
          }
        }
      },
      {
        "phase": 1,
        "weights": {
          "binding": 0.25,
          "recall_med": 0.35,
          "composite_p5": 0.4
        },
        "steps": 18,
        "arm_counts": {
          "binding": 2000,
          "recall_med": 2800,
          "composite_p5": 3200
        },
        "final_loss": 6.799128532409668,
        "loss_curve": [],
        "eval": {
          "binding_L16": {
            "overall": 0.0,
            "prefix": {
              "0": 1.0,
              "1": 0.0,
              "2": 0.0
            },
            "holder_acc": 0.0,
            "value_acc": 0.0
          },
          "recall_easy_L4": {
            "overall": 0.0,
            "prefix": {
              "0": 1.0,
              "1": 0.0,
              "2": 0.0
            },
            "holder_acc": 0.0,
            "value_acc": 0.0
          },
          "recall_med_L8": {
            "overall": 0.0,
            "prefix": {
              "0": 1.0,
              "1": 0.0,
              "2": 0.0
            },
            "holder_acc": 0.0,
            "value_acc": 0.0
          },
          "recall_hard_L16": {
            "overall": 0.0,
            "prefix": {
              "0": 1.0,
              "1": 0.0,
              "2": 0.0
            },
            "holder_acc": 0.0,
            "value_acc": 0.0
          },
          "composite_p5_L16": {
            "overall": 0.0,
            "prefix": {
              "0": 1.0,
              "1": 0.0,
              "2": 0.0
            },
            "holder_acc": 0.0,
            "value_acc": 0.0
          },
          "composite_p16_L16": {
            "overall": 0.0,
            "prefix": {
              "0": 1.0,
              "1": 0.0,
              "2": 0.0
            },
            "holder_acc": 0.0,
            "value_acc": 0.0
          },
          "composite_p16_scaffolded": {
            "scaffolded_value": 0.0
          }
        }
      },
      {
        "phase": 2,
        "weights": {
          "binding": 0.15,
          "recall_hard": 0.25,
          "composite_p5": 0.3,
          "composite_p16": 0.3
        },
        "steps": 18,
        "arm_counts": {
          "binding": 1200,
          "recall_hard": 2000,
          "composite_p5": 2400,
          "composite_p16": 2400
        },
        "final_loss": 6.7234930992126465,
        "loss_curve": [],
        "eval": {
          "binding_L16": {
            "overall": 0.0,
            "prefix": {
              "0": 1.0,
              "1": 0.0,
              "2": 0.0
            },
            "holder_acc": 0.0,
            "value_acc": 0.0
          },
          "recall_easy_L4": {
            "overall": 0.0,
            "prefix": {
              "0": 1.0,
              "1": 0.0,
              "2": 0.0
            },
            "holder_acc": 0.0,
            "value_acc": 0.0
          },
          "recall_med_L8": {
            "overall": 0.0,
            "prefix": {
              "0": 1.0,
              "1": 0.0,
              "2": 0.0
            },
            "holder_acc": 0.0,
            "value_acc": 0.0
          },
          "recall_hard_L16": {
            "overall": 0.0,
            "prefix": {
              "0": 1.0,
              "1": 0.0,
              "2": 0.0
            },
            "holder_acc": 0.0,
            "value_acc": 0.0
          },
          "composite_p5_L16": {
            "overall": 0.0,
            "prefix": {
              "0": 1.0,
              "1": 0.0,
              "2": 0.0
            },
            "holder_acc": 0.0,
            "value_acc": 0.0
          },
          "composite_p16_L16": {
            "overall": 0.0,
            "prefix": {
              "0": 1.0,
              "1": 0.0,
              "2": 0.0
            },
            "holder_acc": 0.0,
            "value_acc": 0.0
          },
          "composite_p16_scaffolded": {
            "scaffolded_value": 0.0
          }
        }
      }
    ],
    "final_loss": 6.7234930992126465,
    "final_eval": {
      "binding_L16": {
        "overall": 0.0,
        "prefix": {
          "0": 1.0,
          "1": 0.0,
          "2": 0.0
        },
        "holder_acc": 0.0,
        "value_acc": 0.0
      },
      "recall_easy_L4": {
        "overall": 0.0,
        "prefix": {
          "0": 1.0,
          "1": 0.0,
          "2": 0.0
        },
        "holder_acc": 0.0,
        "value_acc": 0.0
      },
      "recall_med_L8": {
        "overall": 0.0,
        "prefix": {
          "0": 1.0,
          "1": 0.0,
          "2": 0.0
        },
        "holder_acc": 0.0,
        "value_acc": 0.0
      },
      "recall_hard_L16": {
        "overall": 0.0,
        "prefix": {
          "0": 1.0,
          "1": 0.0,
          "2": 0.0
        },
        "holder_acc": 0.0,
        "value_acc": 0.0
      },
      "composite_p5_L16": {
        "overall": 0.0,
        "prefix": {
          "0": 1.0,
          "1": 0.0,
          "2": 0.0
        },
        "holder_acc": 0.0,
        "value_acc": 0.0
      },
      "composite_p16_L16": {
        "overall": 0.0,
        "prefix": {
          "0": 1.0,
          "1": 0.0,
          "2": 0.0
        },
        "holder_acc": 0.0,
        "value_acc": 0.0
      },
      "composite_p16_scaffolded": {
        "scaffolded_value": 0.0
      }
    },
    "flat": {
      "binding_L16_overall": 0.0,
      "binding_L16_holder": 0.0,
      "binding_L16_value": 0.0,
      "recall_easy_L4_overall": 0.0,
      "recall_easy_L4_holder": 0.0,
      "recall_easy_L4_value": 0.0,
      "recall_med_L8_overall": 0.0,
      "recall_med_L8_holder": 0.0,
      "recall_med_L8_value": 0.0,
      "recall_hard_L16_overall": 0.0,
      "recall_hard_L16_holder": 0.0,
      "recall_hard_L16_value": 0.0,
      "composite_p5_L16_overall": 0.0,
      "composite_p5_L16_holder": 0.0,
      "composite_p5_L16_value": 0.0,
      "composite_p16_L16_overall": 0.0,
      "composite_p16_L16_holder": 0.0,
      "composite_p16_L16_value": 0.0,
      "composite_p16_scaffolded": 0.0
    },
    "arch": "gdp_hybrid"
  }
]
```