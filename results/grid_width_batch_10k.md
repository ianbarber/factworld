# Staged curriculum — architecture/batch grid sweep

grid=d_model:512,768,1024;n_layers:8;batch:32,64;steps:10000;seeds:0

## Per-config final eval (mean over seeds)

| config | bind L16 | recall easy | recall med | recall hard | comp p5 | comp p16 | p16 holder | p16 value | p16 scaffold | final loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| batch32_d_model1024_n_layers8_steps10000_train_n8000 | 0.34 | 0.13 | 0.06 | 0.08 | 0.10 | 0.02 | 0.39 | 0.07 | 0.08 | 0.135 |
| batch32_d_model512_n_layers8_steps10000_train_n8000 | 0.15 | 0.18 | 0.13 | 0.06 | 0.02 | 0.01 | 0.12 | 0.03 | 0.07 | 0.649 |
| batch32_d_model768_n_layers8_steps10000_train_n8000 | 0.60 | 0.17 | 0.15 | 0.02 | 0.08 | 0.01 | 0.65 | 0.03 | 0.07 | 0.271 |
| batch64_d_model1024_n_layers8_steps10000_train_n8000 | 0.86 | 0.15 | 0.12 | 0.05 | 0.14 | 0.03 | 0.85 | 0.03 | 0.07 | 0.099 |
| batch64_d_model512_n_layers8_steps10000_train_n8000 | 0.99 | 0.21 | 0.08 | 0.06 | 0.14 | 0.01 | 1.00 | 0.01 | 0.05 | 0.099 |
| batch64_d_model768_n_layers8_steps10000_train_n8000 | 0.98 | 0.18 | 0.08 | 0.05 | 0.11 | 0.02 | 0.98 | 0.02 | 0.06 | 0.099 |

## Raw results

```json
[
  {
    "cfg": {
      "d_model": 512,
      "n_layers": 8,
      "batch": 32,
      "steps": 10000,
      "seed": 0,
      "train_n": 8000
    },
    "label": "batch32_d_model512_n_layers8_steps10000_train_n8000",
    "stage_records": [
      {
        "phase": 0,
        "weights": {
          "binding": 0.5,
          "recall_easy": 0.5
        },
        "steps": 4000,
        "arm_counts": {
          "binding": 4000,
          "recall_easy": 4000
        },
        "final_loss": 0.870580792427063,
        "loss_curve": [
          [
            200,
            0.9488390684127808
          ],
          [
            400,
            1.5088233947753906
          ],
          [
            600,
            1.0364506244659424
          ],
          [
            800,
            0.9442289471626282
          ],
          [
            1000,
            0.9492459297180176
          ],
          [
            1200,
            1.592237114906311
          ],
          [
            1400,
            0.9270098805427551
          ],
          [
            1600,
            0.9904935956001282
          ],
          [
            1800,
            0.8883412480354309
          ],
          [
            2000,
            0.9904739856719971
          ],
          [
            2200,
            0.8750154376029968
          ],
          [
            2400,
            0.9343271255493164
          ],
          [
            2600,
            0.9620609283447266
          ],
          [
            2800,
            0.9331890940666199
          ],
          [
            3000,
            0.913013756275177
          ],
          [
            3200,
            0.8657210469245911
          ],
          [
            3400,
            0.9624353647232056
          ],
          [
            3600,
            0.8964439034461975
          ],
          [
            3800,
            0.3651387393474579
          ],
          [
            4000,
            0.870580792427063
          ]
        ],
        "eval": {
          "binding_L16": {
            "overall": 0.02,
            "prefix": {
              "0": 0.98,
              "1": 0.02,
              "2": 0.0
            },
            "holder_acc": 0.02,
            "value_acc": 0.0
          },
          "recall_easy_L4": {
            "overall": 0.24,
            "prefix": {
              "0": 0.76,
              "1": 0.24,
              "2": 0.0
            },
            "holder_acc": 0.24,
            "value_acc": 0.0
          },
          "recall_med_L8": {
            "overall": 0.13,
            "prefix": {
              "0": 0.87,
              "1": 0.13,
              "2": 0.0
            },
            "holder_acc": 0.13,
            "value_acc": 0.0
          },
          "recall_hard_L16": {
            "overall": 0.05,
            "prefix": {
              "0": 0.95,
              "1": 0.05,
              "2": 0.0
            },
            "holder_acc": 0.05,
            "value_acc": 0.0
          },
          "composite_p5_L16": {
            "overall": 0.0,
            "prefix": {
              "0": 0.98,
              "1": 0.02,
              "2": 0.0
            },
            "holder_acc": 0.02,
            "value_acc": 0.0
          },
          "composite_p16_L16": {
            "overall": 0.0,
            "prefix": {
              "0": 0.96,
              "1": 0.04,
              "2": 0.0
            },
            "holder_acc": 0.04,
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
        "steps": 3000,
        "arm_counts": {
          "binding": 2000,
          "recall_med": 2800,
          "composite_p5": 3200
        },
        "final_loss": 0.2329903095960617,
        "loss_curve": [
          [
            200,
            1.4176528453826904
          ],
          [
            400,
            0.8558782935142517
          ],
          [
            600,
            0.8995969295501709
          ],
          [
            800,
            0.7894837260246277
          ],
          [
            1000,
            1.0108616352081299
          ],
          [
            1200,
            0.9263446927070618
          ],
          [
            1400,
            0.4789455831050873
          ],
          [
            1600,
            0.7426804900169373
          ],
          [
            1800,
            1.053246021270752
          ],
          [
            2000,
            0.7299100756645203
          ],
          [
            2200,
            0.8636095523834229
          ],
          [
            2400,
            0.18316872417926788
          ],
          [
            2600,
            0.34542739391326904
          ],
          [
            2800,
            0.29625895619392395
          ],
          [
            3000,
            0.2329903095960617
          ]
        ],
        "eval": {
          "binding_L16": {
            "overall": 0.08,
            "prefix": {
              "0": 0.92,
              "1": 0.08,
              "2": 0.0
            },
            "holder_acc": 0.08,
            "value_acc": 0.0
          },
          "recall_easy_L4": {
            "overall": 0.22,
            "prefix": {
              "0": 0.78,
              "1": 0.22,
              "2": 0.0
            },
            "holder_acc": 0.22,
            "value_acc": 0.0
          },
          "recall_med_L8": {
            "overall": 0.1,
            "prefix": {
              "0": 0.9,
              "1": 0.1,
              "2": 0.0
            },
            "holder_acc": 0.1,
            "value_acc": 0.0
          },
          "recall_hard_L16": {
            "overall": 0.12,
            "prefix": {
              "0": 0.88,
              "1": 0.12,
              "2": 0.0
            },
            "holder_acc": 0.12,
            "value_acc": 0.0
          },
          "composite_p5_L16": {
            "overall": 0.05,
            "prefix": {
              "0": 0.79,
              "1": 0.16,
              "2": 0.05
            },
            "holder_acc": 0.21,
            "value_acc": 0.22
          },
          "composite_p16_L16": {
            "overall": 0.0,
            "prefix": {
              "0": 0.93,
              "1": 0.07,
              "2": 0.0
            },
            "holder_acc": 0.07,
            "value_acc": 0.02
          },
          "composite_p16_scaffolded": {
            "scaffolded_value": 0.03
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
        "steps": 3000,
        "arm_counts": {
          "binding": 1200,
          "recall_hard": 2000,
          "composite_p5": 2400,
          "composite_p16": 2400
        },
        "final_loss": 0.6485147476196289,
        "loss_curve": [
          [
            200,
            1.933579683303833
          ],
          [
            400,
            0.5812880992889404
          ],
          [
            600,
            0.892096221446991
          ],
          [
            800,
            0.4074552357196808
          ],
          [
            1000,
            0.8901198506355286
          ],
          [
            1200,
            1.0384711027145386
          ],
          [
            1400,
            0.23430100083351135
          ],
          [
            1600,
            0.6538565754890442
          ],
          [
            1800,
            0.9566776752471924
          ],
          [
            2000,
            0.6175661683082581
          ],
          [
            2200,
            0.4615200459957123
          ],
          [
            2400,
            0.09691289812326431
          ],
          [
            2600,
            0.08276354521512985
          ],
          [
            2800,
            0.07355358451604843
          ],
          [
            3000,
            0.6485147476196289
          ]
        ],
        "eval": {
          "binding_L16": {
            "overall": 0.15,
            "prefix": {
              "0": 0.85,
              "1": 0.15,
              "2": 0.0
            },
            "holder_acc": 0.15,
            "value_acc": 0.0
          },
          "recall_easy_L4": {
            "overall": 0.18,
            "prefix": {
              "0": 0.82,
              "1": 0.18,
              "2": 0.0
            },
            "holder_acc": 0.18,
            "value_acc": 0.0
          },
          "recall_med_L8": {
            "overall": 0.13,
            "prefix": {
              "0": 0.87,
              "1": 0.13,
              "2": 0.0
            },
            "holder_acc": 0.13,
            "value_acc": 0.0
          },
          "recall_hard_L16": {
            "overall": 0.06,
            "prefix": {
              "0": 0.94,
              "1": 0.06,
              "2": 0.0
            },
            "holder_acc": 0.06,
            "value_acc": 0.0
          },
          "composite_p5_L16": {
            "overall": 0.02,
            "prefix": {
              "0": 0.77,
              "1": 0.21,
              "2": 0.02
            },
            "holder_acc": 0.23,
            "value_acc": 0.13
          },
          "composite_p16_L16": {
            "overall": 0.01,
            "prefix": {
              "0": 0.88,
              "1": 0.11,
              "2": 0.01
            },
            "holder_acc": 0.12,
            "value_acc": 0.03
          },
          "composite_p16_scaffolded": {
            "scaffolded_value": 0.07
          }
        }
      }
    ],
    "final_loss": 0.6485147476196289,
    "final_eval": {
      "binding_L16": {
        "overall": 0.15,
        "prefix": {
          "0": 0.85,
          "1": 0.15,
          "2": 0.0
        },
        "holder_acc": 0.15,
        "value_acc": 0.0
      },
      "recall_easy_L4": {
        "overall": 0.18,
        "prefix": {
          "0": 0.82,
          "1": 0.18,
          "2": 0.0
        },
        "holder_acc": 0.18,
        "value_acc": 0.0
      },
      "recall_med_L8": {
        "overall": 0.13,
        "prefix": {
          "0": 0.87,
          "1": 0.13,
          "2": 0.0
        },
        "holder_acc": 0.13,
        "value_acc": 0.0
      },
      "recall_hard_L16": {
        "overall": 0.06,
        "prefix": {
          "0": 0.94,
          "1": 0.06,
          "2": 0.0
        },
        "holder_acc": 0.06,
        "value_acc": 0.0
      },
      "composite_p5_L16": {
        "overall": 0.02,
        "prefix": {
          "0": 0.77,
          "1": 0.21,
          "2": 0.02
        },
        "holder_acc": 0.23,
        "value_acc": 0.13
      },
      "composite_p16_L16": {
        "overall": 0.01,
        "prefix": {
          "0": 0.88,
          "1": 0.11,
          "2": 0.01
        },
        "holder_acc": 0.12,
        "value_acc": 0.03
      },
      "composite_p16_scaffolded": {
        "scaffolded_value": 0.07
      }
    },
    "flat": {
      "binding_L16_overall": 0.15,
      "binding_L16_holder": 0.15,
      "binding_L16_value": 0.0,
      "recall_easy_L4_overall": 0.18,
      "recall_easy_L4_holder": 0.18,
      "recall_easy_L4_value": 0.0,
      "recall_med_L8_overall": 0.13,
      "recall_med_L8_holder": 0.13,
      "recall_med_L8_value": 0.0,
      "recall_hard_L16_overall": 0.06,
      "recall_hard_L16_holder": 0.06,
      "recall_hard_L16_value": 0.0,
      "composite_p5_L16_overall": 0.02,
      "composite_p5_L16_holder": 0.23,
      "composite_p5_L16_value": 0.13,
      "composite_p16_L16_overall": 0.01,
      "composite_p16_L16_holder": 0.12,
      "composite_p16_L16_value": 0.03,
      "composite_p16_scaffolded": 0.07
    },
    "arch": "gdp_hybrid"
  },
  {
    "cfg": {
      "d_model": 512,
      "n_layers": 8,
      "batch": 64,
      "steps": 10000,
      "seed": 0,
      "train_n": 8000
    },
    "label": "batch64_d_model512_n_layers8_steps10000_train_n8000",
    "stage_records": [
      {
        "phase": 0,
        "weights": {
          "binding": 0.5,
          "recall_easy": 0.5
        },
        "steps": 4000,
        "arm_counts": {
          "binding": 4000,
          "recall_easy": 4000
        },
        "final_loss": 0.17079497873783112,
        "loss_curve": [
          [
            200,
            1.880871057510376
          ],
          [
            400,
            0.9717636704444885
          ],
          [
            600,
            1.4296472072601318
          ],
          [
            800,
            1.0815666913986206
          ],
          [
            1000,
            0.8301675915718079
          ],
          [
            1200,
            0.9377765655517578
          ],
          [
            1400,
            0.7994285225868225
          ],
          [
            1600,
            0.47450685501098633
          ],
          [
            1800,
            0.47719404101371765
          ],
          [
            2000,
            0.8993452191352844
          ],
          [
            2200,
            0.8004515171051025
          ],
          [
            2400,
            0.3595012128353119
          ],
          [
            2600,
            0.2752014696598053
          ],
          [
            2800,
            0.2909013032913208
          ],
          [
            3000,
            0.2775219678878784
          ],
          [
            3200,
            0.3629457652568817
          ],
          [
            3400,
            0.30924108624458313
          ],
          [
            3600,
            0.2836630642414093
          ],
          [
            3800,
            0.33614856004714966
          ],
          [
            4000,
            0.17079497873783112
          ]
        ],
        "eval": {
          "binding_L16": {
            "overall": 0.97,
            "prefix": {
              "0": 0.03,
              "1": 0.97,
              "2": 0.0
            },
            "holder_acc": 0.97,
            "value_acc": 0.0
          },
          "recall_easy_L4": {
            "overall": 0.29,
            "prefix": {
              "0": 0.71,
              "1": 0.29,
              "2": 0.0
            },
            "holder_acc": 0.29,
            "value_acc": 0.0
          },
          "recall_med_L8": {
            "overall": 0.03,
            "prefix": {
              "0": 0.97,
              "1": 0.03,
              "2": 0.0
            },
            "holder_acc": 0.03,
            "value_acc": 0.0
          },
          "recall_hard_L16": {
            "overall": 0.06,
            "prefix": {
              "0": 0.94,
              "1": 0.06,
              "2": 0.0
            },
            "holder_acc": 0.06,
            "value_acc": 0.0
          },
          "composite_p5_L16": {
            "overall": 0.0,
            "prefix": {
              "0": 0.09,
              "1": 0.91,
              "2": 0.0
            },
            "holder_acc": 0.91,
            "value_acc": 0.0
          },
          "composite_p16_L16": {
            "overall": 0.0,
            "prefix": {
              "0": 0.14,
              "1": 0.86,
              "2": 0.0
            },
            "holder_acc": 0.86,
            "value_acc": 0.0
          },
          "composite_p16_scaffolded": {
            "scaffolded_value": 0.02
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
        "steps": 3000,
        "arm_counts": {
          "binding": 2000,
          "recall_med": 2800,
          "composite_p5": 3200
        },
        "final_loss": 0.2581251263618469,
        "loss_curve": [
          [
            200,
            1.9398759603500366
          ],
          [
            400,
            0.760064959526062
          ],
          [
            600,
            1.8206274509429932
          ],
          [
            800,
            0.3149396479129791
          ],
          [
            1000,
            0.36033278703689575
          ],
          [
            1200,
            0.27705320715904236
          ],
          [
            1400,
            0.49516692757606506
          ],
          [
            1600,
            0.23732158541679382
          ],
          [
            1800,
            0.1832837164402008
          ],
          [
            2000,
            0.06163148954510689
          ],
          [
            2200,
            0.09751865267753601
          ],
          [
            2400,
            0.1578553169965744
          ],
          [
            2600,
            0.17169082164764404
          ],
          [
            2800,
            0.2544809579849243
          ],
          [
            3000,
            0.2581251263618469
          ]
        ],
        "eval": {
          "binding_L16": {
            "overall": 0.99,
            "prefix": {
              "0": 0.01,
              "1": 0.99,
              "2": 0.0
            },
            "holder_acc": 0.99,
            "value_acc": 0.0
          },
          "recall_easy_L4": {
            "overall": 0.28,
            "prefix": {
              "0": 0.72,
              "1": 0.28,
              "2": 0.0
            },
            "holder_acc": 0.28,
            "value_acc": 0.0
          },
          "recall_med_L8": {
            "overall": 0.09,
            "prefix": {
              "0": 0.91,
              "1": 0.09,
              "2": 0.0
            },
            "holder_acc": 0.09,
            "value_acc": 0.0
          },
          "recall_hard_L16": {
            "overall": 0.08,
            "prefix": {
              "0": 0.92,
              "1": 0.08,
              "2": 0.0
            },
            "holder_acc": 0.08,
            "value_acc": 0.0
          },
          "composite_p5_L16": {
            "overall": 0.16,
            "prefix": {
              "0": 0.02,
              "1": 0.82,
              "2": 0.16
            },
            "holder_acc": 0.98,
            "value_acc": 0.16
          },
          "composite_p16_L16": {
            "overall": 0.02,
            "prefix": {
              "0": 0.0,
              "1": 0.98,
              "2": 0.02
            },
            "holder_acc": 1.0,
            "value_acc": 0.02
          },
          "composite_p16_scaffolded": {
            "scaffolded_value": 0.04
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
        "steps": 3000,
        "arm_counts": {
          "binding": 1200,
          "recall_hard": 2000,
          "composite_p5": 2400,
          "composite_p16": 2400
        },
        "final_loss": 0.09894367307424545,
        "loss_curve": [
          [
            200,
            2.0429418087005615
          ],
          [
            400,
            0.7545240521430969
          ],
          [
            600,
            0.2282862812280655
          ],
          [
            800,
            0.15858453512191772
          ],
          [
            1000,
            0.15784034132957458
          ],
          [
            1200,
            0.19852343201637268
          ],
          [
            1400,
            0.4953887164592743
          ],
          [
            1600,
            0.12646760046482086
          ],
          [
            1800,
            0.1328628659248352
          ],
          [
            2000,
            0.05264510586857796
          ],
          [
            2200,
            0.058106403797864914
          ],
          [
            2400,
            0.1098707839846611
          ],
          [
            2600,
            0.08134912699460983
          ],
          [
            2800,
            0.09476686269044876
          ],
          [
            3000,
            0.09894367307424545
          ]
        ],
        "eval": {
          "binding_L16": {
            "overall": 0.99,
            "prefix": {
              "0": 0.01,
              "1": 0.99,
              "2": 0.0
            },
            "holder_acc": 0.99,
            "value_acc": 0.0
          },
          "recall_easy_L4": {
            "overall": 0.21,
            "prefix": {
              "0": 0.79,
              "1": 0.21,
              "2": 0.0
            },
            "holder_acc": 0.21,
            "value_acc": 0.0
          },
          "recall_med_L8": {
            "overall": 0.08,
            "prefix": {
              "0": 0.92,
              "1": 0.08,
              "2": 0.0
            },
            "holder_acc": 0.08,
            "value_acc": 0.0
          },
          "recall_hard_L16": {
            "overall": 0.06,
            "prefix": {
              "0": 0.94,
              "1": 0.06,
              "2": 0.0
            },
            "holder_acc": 0.06,
            "value_acc": 0.0
          },
          "composite_p5_L16": {
            "overall": 0.14,
            "prefix": {
              "0": 0.01,
              "1": 0.85,
              "2": 0.14
            },
            "holder_acc": 0.99,
            "value_acc": 0.14
          },
          "composite_p16_L16": {
            "overall": 0.01,
            "prefix": {
              "0": 0.0,
              "1": 0.99,
              "2": 0.01
            },
            "holder_acc": 1.0,
            "value_acc": 0.01
          },
          "composite_p16_scaffolded": {
            "scaffolded_value": 0.05
          }
        }
      }
    ],
    "final_loss": 0.09894367307424545,
    "final_eval": {
      "binding_L16": {
        "overall": 0.99,
        "prefix": {
          "0": 0.01,
          "1": 0.99,
          "2": 0.0
        },
        "holder_acc": 0.99,
        "value_acc": 0.0
      },
      "recall_easy_L4": {
        "overall": 0.21,
        "prefix": {
          "0": 0.79,
          "1": 0.21,
          "2": 0.0
        },
        "holder_acc": 0.21,
        "value_acc": 0.0
      },
      "recall_med_L8": {
        "overall": 0.08,
        "prefix": {
          "0": 0.92,
          "1": 0.08,
          "2": 0.0
        },
        "holder_acc": 0.08,
        "value_acc": 0.0
      },
      "recall_hard_L16": {
        "overall": 0.06,
        "prefix": {
          "0": 0.94,
          "1": 0.06,
          "2": 0.0
        },
        "holder_acc": 0.06,
        "value_acc": 0.0
      },
      "composite_p5_L16": {
        "overall": 0.14,
        "prefix": {
          "0": 0.01,
          "1": 0.85,
          "2": 0.14
        },
        "holder_acc": 0.99,
        "value_acc": 0.14
      },
      "composite_p16_L16": {
        "overall": 0.01,
        "prefix": {
          "0": 0.0,
          "1": 0.99,
          "2": 0.01
        },
        "holder_acc": 1.0,
        "value_acc": 0.01
      },
      "composite_p16_scaffolded": {
        "scaffolded_value": 0.05
      }
    },
    "flat": {
      "binding_L16_overall": 0.99,
      "binding_L16_holder": 0.99,
      "binding_L16_value": 0.0,
      "recall_easy_L4_overall": 0.21,
      "recall_easy_L4_holder": 0.21,
      "recall_easy_L4_value": 0.0,
      "recall_med_L8_overall": 0.08,
      "recall_med_L8_holder": 0.08,
      "recall_med_L8_value": 0.0,
      "recall_hard_L16_overall": 0.06,
      "recall_hard_L16_holder": 0.06,
      "recall_hard_L16_value": 0.0,
      "composite_p5_L16_overall": 0.14,
      "composite_p5_L16_holder": 0.99,
      "composite_p5_L16_value": 0.14,
      "composite_p16_L16_overall": 0.01,
      "composite_p16_L16_holder": 1.0,
      "composite_p16_L16_value": 0.01,
      "composite_p16_scaffolded": 0.05
    },
    "arch": "gdp_hybrid"
  },
  {
    "cfg": {
      "d_model": 768,
      "n_layers": 8,
      "batch": 32,
      "steps": 10000,
      "seed": 0,
      "train_n": 8000
    },
    "label": "batch32_d_model768_n_layers8_steps10000_train_n8000",
    "stage_records": [
      {
        "phase": 0,
        "weights": {
          "binding": 0.5,
          "recall_easy": 0.5
        },
        "steps": 4000,
        "arm_counts": {
          "binding": 4000,
          "recall_easy": 4000
        },
        "final_loss": 0.7661722898483276,
        "loss_curve": [
          [
            200,
            0.9248008728027344
          ],
          [
            400,
            1.4100759029388428
          ],
          [
            600,
            1.0010464191436768
          ],
          [
            800,
            0.9355440139770508
          ],
          [
            1000,
            0.9769262671470642
          ],
          [
            1200,
            1.6279431581497192
          ],
          [
            1400,
            0.951806366443634
          ],
          [
            1600,
            0.9852080941200256
          ],
          [
            1800,
            0.9053913950920105
          ],
          [
            2000,
            0.994550883769989
          ],
          [
            2200,
            0.8856668472290039
          ],
          [
            2400,
            1.0989012718200684
          ],
          [
            2600,
            0.9548275470733643
          ],
          [
            2800,
            0.9225232005119324
          ],
          [
            3000,
            0.8225552439689636
          ],
          [
            3200,
            0.771297037601471
          ],
          [
            3400,
            0.9323915839195251
          ],
          [
            3600,
            0.8138386011123657
          ],
          [
            3800,
            0.36270102858543396
          ],
          [
            4000,
            0.7661722898483276
          ]
        ],
        "eval": {
          "binding_L16": {
            "overall": 0.34,
            "prefix": {
              "0": 0.66,
              "1": 0.34,
              "2": 0.0
            },
            "holder_acc": 0.34,
            "value_acc": 0.0
          },
          "recall_easy_L4": {
            "overall": 0.3,
            "prefix": {
              "0": 0.7,
              "1": 0.3,
              "2": 0.0
            },
            "holder_acc": 0.3,
            "value_acc": 0.0
          },
          "recall_med_L8": {
            "overall": 0.08,
            "prefix": {
              "0": 0.92,
              "1": 0.08,
              "2": 0.0
            },
            "holder_acc": 0.08,
            "value_acc": 0.0
          },
          "recall_hard_L16": {
            "overall": 0.07,
            "prefix": {
              "0": 0.93,
              "1": 0.07,
              "2": 0.0
            },
            "holder_acc": 0.07,
            "value_acc": 0.0
          },
          "composite_p5_L16": {
            "overall": 0.0,
            "prefix": {
              "0": 0.62,
              "1": 0.38,
              "2": 0.0
            },
            "holder_acc": 0.38,
            "value_acc": 0.0
          },
          "composite_p16_L16": {
            "overall": 0.0,
            "prefix": {
              "0": 0.59,
              "1": 0.41,
              "2": 0.0
            },
            "holder_acc": 0.41,
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
        "steps": 3000,
        "arm_counts": {
          "binding": 2000,
          "recall_med": 2800,
          "composite_p5": 3200
        },
        "final_loss": 0.16322913765907288,
        "loss_curve": [
          [
            200,
            1.4316856861114502
          ],
          [
            400,
            0.7384526133537292
          ],
          [
            600,
            1.1318888664245605
          ],
          [
            800,
            0.8564754724502563
          ],
          [
            1000,
            1.0987049341201782
          ],
          [
            1200,
            0.8797877430915833
          ],
          [
            1400,
            0.58782559633255
          ],
          [
            1600,
            0.7927342653274536
          ],
          [
            1800,
            1.1255197525024414
          ],
          [
            2000,
            0.7797664999961853
          ],
          [
            2200,
            0.9316365122795105
          ],
          [
            2400,
            0.19410277903079987
          ],
          [
            2600,
            0.3315730690956116
          ],
          [
            2800,
            0.24342751502990723
          ],
          [
            3000,
            0.16322913765907288
          ]
        ],
        "eval": {
          "binding_L16": {
            "overall": 0.68,
            "prefix": {
              "0": 0.32,
              "1": 0.68,
              "2": 0.0
            },
            "holder_acc": 0.68,
            "value_acc": 0.0
          },
          "recall_easy_L4": {
            "overall": 0.2,
            "prefix": {
              "0": 0.8,
              "1": 0.2,
              "2": 0.0
            },
            "holder_acc": 0.2,
            "value_acc": 0.0
          },
          "recall_med_L8": {
            "overall": 0.09,
            "prefix": {
              "0": 0.91,
              "1": 0.09,
              "2": 0.0
            },
            "holder_acc": 0.09,
            "value_acc": 0.0
          },
          "recall_hard_L16": {
            "overall": 0.07,
            "prefix": {
              "0": 0.93,
              "1": 0.07,
              "2": 0.0
            },
            "holder_acc": 0.07,
            "value_acc": 0.0
          },
          "composite_p5_L16": {
            "overall": 0.11,
            "prefix": {
              "0": 0.33,
              "1": 0.56,
              "2": 0.11
            },
            "holder_acc": 0.67,
            "value_acc": 0.22
          },
          "composite_p16_L16": {
            "overall": 0.04,
            "prefix": {
              "0": 0.38,
              "1": 0.58,
              "2": 0.04
            },
            "holder_acc": 0.62,
            "value_acc": 0.05
          },
          "composite_p16_scaffolded": {
            "scaffolded_value": 0.08
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
        "steps": 3000,
        "arm_counts": {
          "binding": 1200,
          "recall_hard": 2000,
          "composite_p5": 2400,
          "composite_p16": 2400
        },
        "final_loss": 0.2711378037929535,
        "loss_curve": [
          [
            200,
            2.0015130043029785
          ],
          [
            400,
            0.7133907079696655
          ],
          [
            600,
            0.791415274143219
          ],
          [
            800,
            0.3759751617908478
          ],
          [
            1000,
            0.9344094395637512
          ],
          [
            1200,
            0.9944691061973572
          ],
          [
            1400,
            0.16446517407894135
          ],
          [
            1600,
            0.5649224519729614
          ],
          [
            1800,
            0.9444172382354736
          ],
          [
            2000,
            0.5016686320304871
          ],
          [
            2200,
            0.45817112922668457
          ],
          [
            2400,
            0.09743312746286392
          ],
          [
            2600,
            0.05596080422401428
          ],
          [
            2800,
            0.07261088490486145
          ],
          [
            3000,
            0.2711378037929535
          ]
        ],
        "eval": {
          "binding_L16": {
            "overall": 0.6,
            "prefix": {
              "0": 0.4,
              "1": 0.6,
              "2": 0.0
            },
            "holder_acc": 0.6,
            "value_acc": 0.0
          },
          "recall_easy_L4": {
            "overall": 0.17,
            "prefix": {
              "0": 0.83,
              "1": 0.17,
              "2": 0.0
            },
            "holder_acc": 0.17,
            "value_acc": 0.0
          },
          "recall_med_L8": {
            "overall": 0.15,
            "prefix": {
              "0": 0.85,
              "1": 0.15,
              "2": 0.0
            },
            "holder_acc": 0.15,
            "value_acc": 0.0
          },
          "recall_hard_L16": {
            "overall": 0.02,
            "prefix": {
              "0": 0.98,
              "1": 0.02,
              "2": 0.0
            },
            "holder_acc": 0.02,
            "value_acc": 0.0
          },
          "composite_p5_L16": {
            "overall": 0.08,
            "prefix": {
              "0": 0.29,
              "1": 0.63,
              "2": 0.08
            },
            "holder_acc": 0.71,
            "value_acc": 0.13
          },
          "composite_p16_L16": {
            "overall": 0.01,
            "prefix": {
              "0": 0.35,
              "1": 0.64,
              "2": 0.01
            },
            "holder_acc": 0.65,
            "value_acc": 0.03
          },
          "composite_p16_scaffolded": {
            "scaffolded_value": 0.07
          }
        }
      }
    ],
    "final_loss": 0.2711378037929535,
    "final_eval": {
      "binding_L16": {
        "overall": 0.6,
        "prefix": {
          "0": 0.4,
          "1": 0.6,
          "2": 0.0
        },
        "holder_acc": 0.6,
        "value_acc": 0.0
      },
      "recall_easy_L4": {
        "overall": 0.17,
        "prefix": {
          "0": 0.83,
          "1": 0.17,
          "2": 0.0
        },
        "holder_acc": 0.17,
        "value_acc": 0.0
      },
      "recall_med_L8": {
        "overall": 0.15,
        "prefix": {
          "0": 0.85,
          "1": 0.15,
          "2": 0.0
        },
        "holder_acc": 0.15,
        "value_acc": 0.0
      },
      "recall_hard_L16": {
        "overall": 0.02,
        "prefix": {
          "0": 0.98,
          "1": 0.02,
          "2": 0.0
        },
        "holder_acc": 0.02,
        "value_acc": 0.0
      },
      "composite_p5_L16": {
        "overall": 0.08,
        "prefix": {
          "0": 0.29,
          "1": 0.63,
          "2": 0.08
        },
        "holder_acc": 0.71,
        "value_acc": 0.13
      },
      "composite_p16_L16": {
        "overall": 0.01,
        "prefix": {
          "0": 0.35,
          "1": 0.64,
          "2": 0.01
        },
        "holder_acc": 0.65,
        "value_acc": 0.03
      },
      "composite_p16_scaffolded": {
        "scaffolded_value": 0.07
      }
    },
    "flat": {
      "binding_L16_overall": 0.6,
      "binding_L16_holder": 0.6,
      "binding_L16_value": 0.0,
      "recall_easy_L4_overall": 0.17,
      "recall_easy_L4_holder": 0.17,
      "recall_easy_L4_value": 0.0,
      "recall_med_L8_overall": 0.15,
      "recall_med_L8_holder": 0.15,
      "recall_med_L8_value": 0.0,
      "recall_hard_L16_overall": 0.02,
      "recall_hard_L16_holder": 0.02,
      "recall_hard_L16_value": 0.0,
      "composite_p5_L16_overall": 0.08,
      "composite_p5_L16_holder": 0.71,
      "composite_p5_L16_value": 0.13,
      "composite_p16_L16_overall": 0.01,
      "composite_p16_L16_holder": 0.65,
      "composite_p16_L16_value": 0.03,
      "composite_p16_scaffolded": 0.07
    },
    "arch": "gdp_hybrid"
  },
  {
    "cfg": {
      "d_model": 768,
      "n_layers": 8,
      "batch": 64,
      "steps": 10000,
      "seed": 0,
      "train_n": 8000
    },
    "label": "batch64_d_model768_n_layers8_steps10000_train_n8000",
    "stage_records": [
      {
        "phase": 0,
        "weights": {
          "binding": 0.5,
          "recall_easy": 0.5
        },
        "steps": 4000,
        "arm_counts": {
          "binding": 4000,
          "recall_easy": 4000
        },
        "final_loss": 0.14412254095077515,
        "loss_curve": [
          [
            200,
            1.7690083980560303
          ],
          [
            400,
            1.0011941194534302
          ],
          [
            600,
            1.3943464756011963
          ],
          [
            800,
            0.9604106545448303
          ],
          [
            1000,
            0.8808813095092773
          ],
          [
            1200,
            0.9675092101097107
          ],
          [
            1400,
            0.865117609500885
          ],
          [
            1600,
            0.49959495663642883
          ],
          [
            1800,
            0.47540345788002014
          ],
          [
            2000,
            0.9142844676971436
          ],
          [
            2200,
            0.8157981038093567
          ],
          [
            2400,
            0.3764544129371643
          ],
          [
            2600,
            0.2822389304637909
          ],
          [
            2800,
            0.29603323340415955
          ],
          [
            3000,
            0.27861326932907104
          ],
          [
            3200,
            0.28058239817619324
          ],
          [
            3400,
            0.30981025099754333
          ],
          [
            3600,
            0.2870531380176544
          ],
          [
            3800,
            0.3351430296897888
          ],
          [
            4000,
            0.14412254095077515
          ]
        ],
        "eval": {
          "binding_L16": {
            "overall": 0.92,
            "prefix": {
              "0": 0.08,
              "1": 0.92,
              "2": 0.0
            },
            "holder_acc": 0.92,
            "value_acc": 0.0
          },
          "recall_easy_L4": {
            "overall": 0.21,
            "prefix": {
              "0": 0.79,
              "1": 0.21,
              "2": 0.0
            },
            "holder_acc": 0.21,
            "value_acc": 0.0
          },
          "recall_med_L8": {
            "overall": 0.14,
            "prefix": {
              "0": 0.86,
              "1": 0.14,
              "2": 0.0
            },
            "holder_acc": 0.14,
            "value_acc": 0.0
          },
          "recall_hard_L16": {
            "overall": 0.09,
            "prefix": {
              "0": 0.91,
              "1": 0.09,
              "2": 0.0
            },
            "holder_acc": 0.09,
            "value_acc": 0.0
          },
          "composite_p5_L16": {
            "overall": 0.0,
            "prefix": {
              "0": 0.12,
              "1": 0.88,
              "2": 0.0
            },
            "holder_acc": 0.88,
            "value_acc": 0.0
          },
          "composite_p16_L16": {
            "overall": 0.0,
            "prefix": {
              "0": 0.1,
              "1": 0.9,
              "2": 0.0
            },
            "holder_acc": 0.9,
            "value_acc": 0.0
          },
          "composite_p16_scaffolded": {
            "scaffolded_value": 0.01
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
        "steps": 3000,
        "arm_counts": {
          "binding": 2000,
          "recall_med": 2800,
          "composite_p5": 3200
        },
        "final_loss": 0.25824013352394104,
        "loss_curve": [
          [
            200,
            1.9187800884246826
          ],
          [
            400,
            0.71478670835495
          ],
          [
            600,
            1.7907962799072266
          ],
          [
            800,
            0.31150147318840027
          ],
          [
            1000,
            0.33797863125801086
          ],
          [
            1200,
            0.27738162875175476
          ],
          [
            1400,
            0.478098601102829
          ],
          [
            1600,
            0.2353786677122116
          ],
          [
            1800,
            0.17794306576251984
          ],
          [
            2000,
            0.061525892466306686
          ],
          [
            2200,
            0.09575481712818146
          ],
          [
            2400,
            0.15835343301296234
          ],
          [
            2600,
            0.17117644846439362
          ],
          [
            2800,
            0.2541905641555786
          ],
          [
            3000,
            0.25824013352394104
          ]
        ],
        "eval": {
          "binding_L16": {
            "overall": 0.96,
            "prefix": {
              "0": 0.04,
              "1": 0.96,
              "2": 0.0
            },
            "holder_acc": 0.96,
            "value_acc": 0.0
          },
          "recall_easy_L4": {
            "overall": 0.29,
            "prefix": {
              "0": 0.71,
              "1": 0.29,
              "2": 0.0
            },
            "holder_acc": 0.29,
            "value_acc": 0.0
          },
          "recall_med_L8": {
            "overall": 0.14,
            "prefix": {
              "0": 0.86,
              "1": 0.14,
              "2": 0.0
            },
            "holder_acc": 0.14,
            "value_acc": 0.0
          },
          "recall_hard_L16": {
            "overall": 0.05,
            "prefix": {
              "0": 0.95,
              "1": 0.05,
              "2": 0.0
            },
            "holder_acc": 0.05,
            "value_acc": 0.0
          },
          "composite_p5_L16": {
            "overall": 0.06,
            "prefix": {
              "0": 0.02,
              "1": 0.92,
              "2": 0.06
            },
            "holder_acc": 0.98,
            "value_acc": 0.07
          },
          "composite_p16_L16": {
            "overall": 0.02,
            "prefix": {
              "0": 0.06,
              "1": 0.92,
              "2": 0.02
            },
            "holder_acc": 0.94,
            "value_acc": 0.02
          },
          "composite_p16_scaffolded": {
            "scaffolded_value": 0.04
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
        "steps": 3000,
        "arm_counts": {
          "binding": 1200,
          "recall_hard": 2000,
          "composite_p5": 2400,
          "composite_p16": 2400
        },
        "final_loss": 0.09900560975074768,
        "loss_curve": [
          [
            200,
            2.031506299972534
          ],
          [
            400,
            0.638587236404419
          ],
          [
            600,
            0.21371124684810638
          ],
          [
            800,
            0.15326032042503357
          ],
          [
            1000,
            0.15339164435863495
          ],
          [
            1200,
            0.165749654173851
          ],
          [
            1400,
            0.4461834728717804
          ],
          [
            1600,
            0.12732593715190887
          ],
          [
            1800,
            0.12292788922786713
          ],
          [
            2000,
            0.05212913453578949
          ],
          [
            2200,
            0.05825209245085716
          ],
          [
            2400,
            0.10978692770004272
          ],
          [
            2600,
            0.08185876905918121
          ],
          [
            2800,
            0.0942937508225441
          ],
          [
            3000,
            0.09900560975074768
          ]
        ],
        "eval": {
          "binding_L16": {
            "overall": 0.98,
            "prefix": {
              "0": 0.02,
              "1": 0.98,
              "2": 0.0
            },
            "holder_acc": 0.98,
            "value_acc": 0.0
          },
          "recall_easy_L4": {
            "overall": 0.18,
            "prefix": {
              "0": 0.82,
              "1": 0.18,
              "2": 0.0
            },
            "holder_acc": 0.18,
            "value_acc": 0.0
          },
          "recall_med_L8": {
            "overall": 0.08,
            "prefix": {
              "0": 0.92,
              "1": 0.08,
              "2": 0.0
            },
            "holder_acc": 0.08,
            "value_acc": 0.0
          },
          "recall_hard_L16": {
            "overall": 0.05,
            "prefix": {
              "0": 0.95,
              "1": 0.05,
              "2": 0.0
            },
            "holder_acc": 0.05,
            "value_acc": 0.0
          },
          "composite_p5_L16": {
            "overall": 0.11,
            "prefix": {
              "0": 0.02,
              "1": 0.87,
              "2": 0.11
            },
            "holder_acc": 0.98,
            "value_acc": 0.11
          },
          "composite_p16_L16": {
            "overall": 0.02,
            "prefix": {
              "0": 0.02,
              "1": 0.96,
              "2": 0.02
            },
            "holder_acc": 0.98,
            "value_acc": 0.02
          },
          "composite_p16_scaffolded": {
            "scaffolded_value": 0.06
          }
        }
      }
    ],
    "final_loss": 0.09900560975074768,
    "final_eval": {
      "binding_L16": {
        "overall": 0.98,
        "prefix": {
          "0": 0.02,
          "1": 0.98,
          "2": 0.0
        },
        "holder_acc": 0.98,
        "value_acc": 0.0
      },
      "recall_easy_L4": {
        "overall": 0.18,
        "prefix": {
          "0": 0.82,
          "1": 0.18,
          "2": 0.0
        },
        "holder_acc": 0.18,
        "value_acc": 0.0
      },
      "recall_med_L8": {
        "overall": 0.08,
        "prefix": {
          "0": 0.92,
          "1": 0.08,
          "2": 0.0
        },
        "holder_acc": 0.08,
        "value_acc": 0.0
      },
      "recall_hard_L16": {
        "overall": 0.05,
        "prefix": {
          "0": 0.95,
          "1": 0.05,
          "2": 0.0
        },
        "holder_acc": 0.05,
        "value_acc": 0.0
      },
      "composite_p5_L16": {
        "overall": 0.11,
        "prefix": {
          "0": 0.02,
          "1": 0.87,
          "2": 0.11
        },
        "holder_acc": 0.98,
        "value_acc": 0.11
      },
      "composite_p16_L16": {
        "overall": 0.02,
        "prefix": {
          "0": 0.02,
          "1": 0.96,
          "2": 0.02
        },
        "holder_acc": 0.98,
        "value_acc": 0.02
      },
      "composite_p16_scaffolded": {
        "scaffolded_value": 0.06
      }
    },
    "flat": {
      "binding_L16_overall": 0.98,
      "binding_L16_holder": 0.98,
      "binding_L16_value": 0.0,
      "recall_easy_L4_overall": 0.18,
      "recall_easy_L4_holder": 0.18,
      "recall_easy_L4_value": 0.0,
      "recall_med_L8_overall": 0.08,
      "recall_med_L8_holder": 0.08,
      "recall_med_L8_value": 0.0,
      "recall_hard_L16_overall": 0.05,
      "recall_hard_L16_holder": 0.05,
      "recall_hard_L16_value": 0.0,
      "composite_p5_L16_overall": 0.11,
      "composite_p5_L16_holder": 0.98,
      "composite_p5_L16_value": 0.11,
      "composite_p16_L16_overall": 0.02,
      "composite_p16_L16_holder": 0.98,
      "composite_p16_L16_value": 0.02,
      "composite_p16_scaffolded": 0.06
    },
    "arch": "gdp_hybrid"
  },
  {
    "cfg": {
      "d_model": 1024,
      "n_layers": 8,
      "batch": 32,
      "steps": 10000,
      "seed": 0,
      "train_n": 8000
    },
    "label": "batch32_d_model1024_n_layers8_steps10000_train_n8000",
    "stage_records": [
      {
        "phase": 0,
        "weights": {
          "binding": 0.5,
          "recall_easy": 0.5
        },
        "steps": 4000,
        "arm_counts": {
          "binding": 4000,
          "recall_easy": 4000
        },
        "final_loss": 0.7730111479759216,
        "loss_curve": [
          [
            200,
            0.9028429388999939
          ],
          [
            400,
            1.4015153646469116
          ],
          [
            600,
            1.1061385869979858
          ],
          [
            800,
            0.9382502436637878
          ],
          [
            1000,
            0.9928751587867737
          ],
          [
            1200,
            1.6676274538040161
          ],
          [
            1400,
            0.9834080934524536
          ],
          [
            1600,
            0.9943079352378845
          ],
          [
            1800,
            0.9080087542533875
          ],
          [
            2000,
            0.9944652318954468
          ],
          [
            2200,
            0.8830904364585876
          ],
          [
            2400,
            1.201321005821228
          ],
          [
            2600,
            0.9589773416519165
          ],
          [
            2800,
            0.9261142611503601
          ],
          [
            3000,
            0.8555065989494324
          ],
          [
            3200,
            0.8029008507728577
          ],
          [
            3400,
            0.9405394196510315
          ],
          [
            3600,
            0.8210955858230591
          ],
          [
            3800,
            0.3702396750450134
          ],
          [
            4000,
            0.7730111479759216
          ]
        ],
        "eval": {
          "binding_L16": {
            "overall": 0.25,
            "prefix": {
              "0": 0.75,
              "1": 0.25,
              "2": 0.0
            },
            "holder_acc": 0.25,
            "value_acc": 0.0
          },
          "recall_easy_L4": {
            "overall": 0.22,
            "prefix": {
              "0": 0.78,
              "1": 0.22,
              "2": 0.0
            },
            "holder_acc": 0.22,
            "value_acc": 0.0
          },
          "recall_med_L8": {
            "overall": 0.08,
            "prefix": {
              "0": 0.92,
              "1": 0.08,
              "2": 0.0
            },
            "holder_acc": 0.08,
            "value_acc": 0.0
          },
          "recall_hard_L16": {
            "overall": 0.1,
            "prefix": {
              "0": 0.9,
              "1": 0.1,
              "2": 0.0
            },
            "holder_acc": 0.1,
            "value_acc": 0.0
          },
          "composite_p5_L16": {
            "overall": 0.0,
            "prefix": {
              "0": 0.76,
              "1": 0.24,
              "2": 0.0
            },
            "holder_acc": 0.24,
            "value_acc": 0.0
          },
          "composite_p16_L16": {
            "overall": 0.0,
            "prefix": {
              "0": 0.78,
              "1": 0.22,
              "2": 0.0
            },
            "holder_acc": 0.22,
            "value_acc": 0.0
          },
          "composite_p16_scaffolded": {
            "scaffolded_value": 0.01
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
        "steps": 3000,
        "arm_counts": {
          "binding": 2000,
          "recall_med": 2800,
          "composite_p5": 3200
        },
        "final_loss": 0.22809898853302002,
        "loss_curve": [
          [
            200,
            1.3776828050613403
          ],
          [
            400,
            0.749081015586853
          ],
          [
            600,
            1.23581862449646
          ],
          [
            800,
            0.9043197631835938
          ],
          [
            1000,
            1.1628005504608154
          ],
          [
            1200,
            0.9034152030944824
          ],
          [
            1400,
            0.7888447642326355
          ],
          [
            1600,
            0.8660222887992859
          ],
          [
            1800,
            1.237945795059204
          ],
          [
            2000,
            0.8344503045082092
          ],
          [
            2200,
            0.9502921104431152
          ],
          [
            2400,
            0.2861958146095276
          ],
          [
            2600,
            0.3756810128688812
          ],
          [
            2800,
            0.2948468327522278
          ],
          [
            3000,
            0.22809898853302002
          ]
        ],
        "eval": {
          "binding_L16": {
            "overall": 0.28,
            "prefix": {
              "0": 0.72,
              "1": 0.28,
              "2": 0.0
            },
            "holder_acc": 0.28,
            "value_acc": 0.0
          },
          "recall_easy_L4": {
            "overall": 0.18,
            "prefix": {
              "0": 0.82,
              "1": 0.18,
              "2": 0.0
            },
            "holder_acc": 0.18,
            "value_acc": 0.0
          },
          "recall_med_L8": {
            "overall": 0.15,
            "prefix": {
              "0": 0.85,
              "1": 0.15,
              "2": 0.0
            },
            "holder_acc": 0.15,
            "value_acc": 0.0
          },
          "recall_hard_L16": {
            "overall": 0.06,
            "prefix": {
              "0": 0.94,
              "1": 0.06,
              "2": 0.0
            },
            "holder_acc": 0.06,
            "value_acc": 0.0
          },
          "composite_p5_L16": {
            "overall": 0.07,
            "prefix": {
              "0": 0.67,
              "1": 0.26,
              "2": 0.07
            },
            "holder_acc": 0.33,
            "value_acc": 0.28
          },
          "composite_p16_L16": {
            "overall": 0.01,
            "prefix": {
              "0": 0.69,
              "1": 0.3,
              "2": 0.01
            },
            "holder_acc": 0.31,
            "value_acc": 0.08
          },
          "composite_p16_scaffolded": {
            "scaffolded_value": 0.11
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
        "steps": 3000,
        "arm_counts": {
          "binding": 1200,
          "recall_hard": 2000,
          "composite_p5": 2400,
          "composite_p16": 2400
        },
        "final_loss": 0.13486850261688232,
        "loss_curve": [
          [
            200,
            1.9468871355056763
          ],
          [
            400,
            0.7880344986915588
          ],
          [
            600,
            0.5490372776985168
          ],
          [
            800,
            0.39253735542297363
          ],
          [
            1000,
            1.0910650491714478
          ],
          [
            1200,
            1.0586897134780884
          ],
          [
            1400,
            0.18872994184494019
          ],
          [
            1600,
            0.6406054496765137
          ],
          [
            1800,
            1.025192379951477
          ],
          [
            2000,
            0.5447459816932678
          ],
          [
            2200,
            0.4917515218257904
          ],
          [
            2400,
            0.10014010220766068
          ],
          [
            2600,
            0.057351868599653244
          ],
          [
            2800,
            0.07311225682497025
          ],
          [
            3000,
            0.13486850261688232
          ]
        ],
        "eval": {
          "binding_L16": {
            "overall": 0.34,
            "prefix": {
              "0": 0.66,
              "1": 0.34,
              "2": 0.0
            },
            "holder_acc": 0.34,
            "value_acc": 0.0
          },
          "recall_easy_L4": {
            "overall": 0.13,
            "prefix": {
              "0": 0.87,
              "1": 0.13,
              "2": 0.0
            },
            "holder_acc": 0.13,
            "value_acc": 0.0
          },
          "recall_med_L8": {
            "overall": 0.06,
            "prefix": {
              "0": 0.94,
              "1": 0.06,
              "2": 0.0
            },
            "holder_acc": 0.06,
            "value_acc": 0.0
          },
          "recall_hard_L16": {
            "overall": 0.08,
            "prefix": {
              "0": 0.92,
              "1": 0.08,
              "2": 0.0
            },
            "holder_acc": 0.08,
            "value_acc": 0.0
          },
          "composite_p5_L16": {
            "overall": 0.1,
            "prefix": {
              "0": 0.63,
              "1": 0.27,
              "2": 0.1
            },
            "holder_acc": 0.37,
            "value_acc": 0.22
          },
          "composite_p16_L16": {
            "overall": 0.02,
            "prefix": {
              "0": 0.61,
              "1": 0.37,
              "2": 0.02
            },
            "holder_acc": 0.39,
            "value_acc": 0.07
          },
          "composite_p16_scaffolded": {
            "scaffolded_value": 0.08
          }
        }
      }
    ],
    "final_loss": 0.13486850261688232,
    "final_eval": {
      "binding_L16": {
        "overall": 0.34,
        "prefix": {
          "0": 0.66,
          "1": 0.34,
          "2": 0.0
        },
        "holder_acc": 0.34,
        "value_acc": 0.0
      },
      "recall_easy_L4": {
        "overall": 0.13,
        "prefix": {
          "0": 0.87,
          "1": 0.13,
          "2": 0.0
        },
        "holder_acc": 0.13,
        "value_acc": 0.0
      },
      "recall_med_L8": {
        "overall": 0.06,
        "prefix": {
          "0": 0.94,
          "1": 0.06,
          "2": 0.0
        },
        "holder_acc": 0.06,
        "value_acc": 0.0
      },
      "recall_hard_L16": {
        "overall": 0.08,
        "prefix": {
          "0": 0.92,
          "1": 0.08,
          "2": 0.0
        },
        "holder_acc": 0.08,
        "value_acc": 0.0
      },
      "composite_p5_L16": {
        "overall": 0.1,
        "prefix": {
          "0": 0.63,
          "1": 0.27,
          "2": 0.1
        },
        "holder_acc": 0.37,
        "value_acc": 0.22
      },
      "composite_p16_L16": {
        "overall": 0.02,
        "prefix": {
          "0": 0.61,
          "1": 0.37,
          "2": 0.02
        },
        "holder_acc": 0.39,
        "value_acc": 0.07
      },
      "composite_p16_scaffolded": {
        "scaffolded_value": 0.08
      }
    },
    "flat": {
      "binding_L16_overall": 0.34,
      "binding_L16_holder": 0.34,
      "binding_L16_value": 0.0,
      "recall_easy_L4_overall": 0.13,
      "recall_easy_L4_holder": 0.13,
      "recall_easy_L4_value": 0.0,
      "recall_med_L8_overall": 0.06,
      "recall_med_L8_holder": 0.06,
      "recall_med_L8_value": 0.0,
      "recall_hard_L16_overall": 0.08,
      "recall_hard_L16_holder": 0.08,
      "recall_hard_L16_value": 0.0,
      "composite_p5_L16_overall": 0.1,
      "composite_p5_L16_holder": 0.37,
      "composite_p5_L16_value": 0.22,
      "composite_p16_L16_overall": 0.02,
      "composite_p16_L16_holder": 0.39,
      "composite_p16_L16_value": 0.07,
      "composite_p16_scaffolded": 0.08
    },
    "arch": "gdp_hybrid"
  },
  {
    "cfg": {
      "d_model": 1024,
      "n_layers": 8,
      "batch": 64,
      "steps": 10000,
      "seed": 0,
      "train_n": 8000
    },
    "label": "batch64_d_model1024_n_layers8_steps10000_train_n8000",
    "stage_records": [
      {
        "phase": 0,
        "weights": {
          "binding": 0.5,
          "recall_easy": 0.5
        },
        "steps": 4000,
        "arm_counts": {
          "binding": 4000,
          "recall_easy": 4000
        },
        "final_loss": 0.6439825296401978,
        "loss_curve": [
          [
            200,
            1.8036340475082397
          ],
          [
            400,
            0.9828052520751953
          ],
          [
            600,
            1.417278528213501
          ],
          [
            800,
            1.2782092094421387
          ],
          [
            1000,
            0.8772301077842712
          ],
          [
            1200,
            1.2183947563171387
          ],
          [
            1400,
            0.9013046622276306
          ],
          [
            1600,
            0.6257579922676086
          ],
          [
            1800,
            0.6432466506958008
          ],
          [
            2000,
            0.9293533563613892
          ],
          [
            2200,
            0.866156280040741
          ],
          [
            2400,
            0.3928644359111786
          ],
          [
            2600,
            0.28514736890792847
          ],
          [
            2800,
            0.2953457236289978
          ],
          [
            3000,
            0.2772156596183777
          ],
          [
            3200,
            0.7664687633514404
          ],
          [
            3400,
            0.31073546409606934
          ],
          [
            3600,
            0.29311123490333557
          ],
          [
            3800,
            0.33520838618278503
          ],
          [
            4000,
            0.6439825296401978
          ]
        ],
        "eval": {
          "binding_L16": {
            "overall": 0.56,
            "prefix": {
              "0": 0.44,
              "1": 0.56,
              "2": 0.0
            },
            "holder_acc": 0.56,
            "value_acc": 0.0
          },
          "recall_easy_L4": {
            "overall": 0.28,
            "prefix": {
              "0": 0.72,
              "1": 0.28,
              "2": 0.0
            },
            "holder_acc": 0.28,
            "value_acc": 0.0
          },
          "recall_med_L8": {
            "overall": 0.14,
            "prefix": {
              "0": 0.86,
              "1": 0.14,
              "2": 0.0
            },
            "holder_acc": 0.14,
            "value_acc": 0.0
          },
          "recall_hard_L16": {
            "overall": 0.05,
            "prefix": {
              "0": 0.95,
              "1": 0.05,
              "2": 0.0
            },
            "holder_acc": 0.05,
            "value_acc": 0.0
          },
          "composite_p5_L16": {
            "overall": 0.0,
            "prefix": {
              "0": 0.49,
              "1": 0.51,
              "2": 0.0
            },
            "holder_acc": 0.51,
            "value_acc": 0.0
          },
          "composite_p16_L16": {
            "overall": 0.0,
            "prefix": {
              "0": 0.44,
              "1": 0.56,
              "2": 0.0
            },
            "holder_acc": 0.56,
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
        "steps": 3000,
        "arm_counts": {
          "binding": 2000,
          "recall_med": 2800,
          "composite_p5": 3200
        },
        "final_loss": 0.25685733556747437,
        "loss_curve": [
          [
            200,
            1.9214280843734741
          ],
          [
            400,
            0.8955382108688354
          ],
          [
            600,
            1.7330008745193481
          ],
          [
            800,
            0.4407825469970703
          ],
          [
            1000,
            0.4759485721588135
          ],
          [
            1200,
            0.46959447860717773
          ],
          [
            1400,
            0.6182534098625183
          ],
          [
            1600,
            0.2375301569700241
          ],
          [
            1800,
            0.17737135291099548
          ],
          [
            2000,
            0.07123606652021408
          ],
          [
            2200,
            0.10822860896587372
          ],
          [
            2400,
            0.1589207649230957
          ],
          [
            2600,
            0.17595423758029938
          ],
          [
            2800,
            0.2565765082836151
          ],
          [
            3000,
            0.25685733556747437
          ]
        ],
        "eval": {
          "binding_L16": {
            "overall": 0.78,
            "prefix": {
              "0": 0.22,
              "1": 0.78,
              "2": 0.0
            },
            "holder_acc": 0.78,
            "value_acc": 0.0
          },
          "recall_easy_L4": {
            "overall": 0.31,
            "prefix": {
              "0": 0.69,
              "1": 0.31,
              "2": 0.0
            },
            "holder_acc": 0.31,
            "value_acc": 0.0
          },
          "recall_med_L8": {
            "overall": 0.13,
            "prefix": {
              "0": 0.87,
              "1": 0.13,
              "2": 0.0
            },
            "holder_acc": 0.13,
            "value_acc": 0.0
          },
          "recall_hard_L16": {
            "overall": 0.07,
            "prefix": {
              "0": 0.93,
              "1": 0.07,
              "2": 0.0
            },
            "holder_acc": 0.07,
            "value_acc": 0.0
          },
          "composite_p5_L16": {
            "overall": 0.19,
            "prefix": {
              "0": 0.17,
              "1": 0.64,
              "2": 0.19
            },
            "holder_acc": 0.83,
            "value_acc": 0.23
          },
          "composite_p16_L16": {
            "overall": 0.05,
            "prefix": {
              "0": 0.28,
              "1": 0.67,
              "2": 0.05
            },
            "holder_acc": 0.72,
            "value_acc": 0.06
          },
          "composite_p16_scaffolded": {
            "scaffolded_value": 0.04
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
        "steps": 3000,
        "arm_counts": {
          "binding": 1200,
          "recall_hard": 2000,
          "composite_p5": 2400,
          "composite_p16": 2400
        },
        "final_loss": 0.09899324178695679,
        "loss_curve": [
          [
            200,
            2.052215337753296
          ],
          [
            400,
            0.8087309002876282
          ],
          [
            600,
            0.22090117633342743
          ],
          [
            800,
            0.1534058302640915
          ],
          [
            1000,
            0.2462860494852066
          ],
          [
            1200,
            0.15556825697422028
          ],
          [
            1400,
            0.4325886070728302
          ],
          [
            1600,
            0.12460888177156448
          ],
          [
            1800,
            0.12475483864545822
          ],
          [
            2000,
            0.05244043841958046
          ],
          [
            2200,
            0.05818380042910576
          ],
          [
            2400,
            0.11062058806419373
          ],
          [
            2600,
            0.08177372068166733
          ],
          [
            2800,
            0.0947910025715828
          ],
          [
            3000,
            0.09899324178695679
          ]
        ],
        "eval": {
          "binding_L16": {
            "overall": 0.86,
            "prefix": {
              "0": 0.14,
              "1": 0.86,
              "2": 0.0
            },
            "holder_acc": 0.86,
            "value_acc": 0.0
          },
          "recall_easy_L4": {
            "overall": 0.15,
            "prefix": {
              "0": 0.85,
              "1": 0.15,
              "2": 0.0
            },
            "holder_acc": 0.15,
            "value_acc": 0.0
          },
          "recall_med_L8": {
            "overall": 0.12,
            "prefix": {
              "0": 0.88,
              "1": 0.12,
              "2": 0.0
            },
            "holder_acc": 0.12,
            "value_acc": 0.0
          },
          "recall_hard_L16": {
            "overall": 0.05,
            "prefix": {
              "0": 0.95,
              "1": 0.05,
              "2": 0.0
            },
            "holder_acc": 0.05,
            "value_acc": 0.0
          },
          "composite_p5_L16": {
            "overall": 0.14,
            "prefix": {
              "0": 0.08,
              "1": 0.78,
              "2": 0.14
            },
            "holder_acc": 0.92,
            "value_acc": 0.17
          },
          "composite_p16_L16": {
            "overall": 0.03,
            "prefix": {
              "0": 0.15,
              "1": 0.82,
              "2": 0.03
            },
            "holder_acc": 0.85,
            "value_acc": 0.03
          },
          "composite_p16_scaffolded": {
            "scaffolded_value": 0.07
          }
        }
      }
    ],
    "final_loss": 0.09899324178695679,
    "final_eval": {
      "binding_L16": {
        "overall": 0.86,
        "prefix": {
          "0": 0.14,
          "1": 0.86,
          "2": 0.0
        },
        "holder_acc": 0.86,
        "value_acc": 0.0
      },
      "recall_easy_L4": {
        "overall": 0.15,
        "prefix": {
          "0": 0.85,
          "1": 0.15,
          "2": 0.0
        },
        "holder_acc": 0.15,
        "value_acc": 0.0
      },
      "recall_med_L8": {
        "overall": 0.12,
        "prefix": {
          "0": 0.88,
          "1": 0.12,
          "2": 0.0
        },
        "holder_acc": 0.12,
        "value_acc": 0.0
      },
      "recall_hard_L16": {
        "overall": 0.05,
        "prefix": {
          "0": 0.95,
          "1": 0.05,
          "2": 0.0
        },
        "holder_acc": 0.05,
        "value_acc": 0.0
      },
      "composite_p5_L16": {
        "overall": 0.14,
        "prefix": {
          "0": 0.08,
          "1": 0.78,
          "2": 0.14
        },
        "holder_acc": 0.92,
        "value_acc": 0.17
      },
      "composite_p16_L16": {
        "overall": 0.03,
        "prefix": {
          "0": 0.15,
          "1": 0.82,
          "2": 0.03
        },
        "holder_acc": 0.85,
        "value_acc": 0.03
      },
      "composite_p16_scaffolded": {
        "scaffolded_value": 0.07
      }
    },
    "flat": {
      "binding_L16_overall": 0.86,
      "binding_L16_holder": 0.86,
      "binding_L16_value": 0.0,
      "recall_easy_L4_overall": 0.15,
      "recall_easy_L4_holder": 0.15,
      "recall_easy_L4_value": 0.0,
      "recall_med_L8_overall": 0.12,
      "recall_med_L8_holder": 0.12,
      "recall_med_L8_value": 0.0,
      "recall_hard_L16_overall": 0.05,
      "recall_hard_L16_holder": 0.05,
      "recall_hard_L16_value": 0.0,
      "composite_p5_L16_overall": 0.14,
      "composite_p5_L16_holder": 0.92,
      "composite_p5_L16_value": 0.17,
      "composite_p16_L16_overall": 0.03,
      "composite_p16_L16_holder": 0.85,
      "composite_p16_L16_value": 0.03,
      "composite_p16_scaffolded": 0.07
    },
    "arch": "gdp_hybrid"
  }
]
```