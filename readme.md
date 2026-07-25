                                           ┌────────────────────────────────────────────┐
                                           │            OFFLINE STAGE (Once)            │
                                           └────────────────────────────────────────────┘

     ┌────────────────────────────────────────────────────────────────────────────────────────────┐
     │                              PUBLIC DATASETS                                               │
     └────────────────────────────────────────────────────────────────────────────────────────────┘

     COCO             TuSimple/CULane           NYU/KITTI          GTSRB
       │                    │                      │                  │
       ▼                    ▼                      ▼                  ▼
   Train/Finetune       Train/Finetune       Pretrained/FT      Train/Finetune
       │                    │                      │                  │
       ▼                    ▼                      ▼                  ▼
    YOLO11 Model      LaneNet Model          MiDaS Model      Traffic Sign Model
       │                    │                      │                  │
       └────────────────────┬──────────────────────┴──────────────────┘
                            │
                            ▼
                   Saved Perception Models (.pt)
                            │
                            │
=========================================================================================
                            │
                            ▼
                  ONLINE DATA COLLECTION STAGE
=========================================================================================

                               ┌────────────────────┐
                               │     MetaDrive      │
                               └────────────────────┘
                                         │
                                 Built-in Expert Driver
                                         │
                                         ▼
                                 Expert Drives Vehicle
                                         │
                                         ▼
                     ┌──────────────────────────────────────┐
                     │     Observation Generation           │
                     └──────────────────────────────────────┘
                                         │
             ┌───────────────────────────┼────────────────────────────┐
             │                           │                            │
             ▼                           ▼                            ▼
        RGB Camera                  Vehicle State             Navigation Command
             │                           │                            │
             └───────────────┬───────────┴───────────────┬────────────┘
                             │                           │
                             ▼                           ▼
                    Load Pretrained Models       Expert Action
                                                 (Steer,Throttle,Brake)
                             │
         ┌───────────────────┼─────────────────────────────┐
         │                   │              │              │
         ▼                   ▼              ▼              ▼
      YOLO11            LaneNet          MiDaS      Traffic Sign Model
         │                   │              │              │
         └───────────────┬───┴──────────────┴───────┬──────┘
                         ▼                          │
               Scene Understanding                 │
                         │                          │
                         ▼                          │
                World State Builder                │
                         │                          │
                         └──────────────┬───────────┘
                                        ▼
                               Dataset Recorder
                                        │
         ┌──────────────────────────────┼─────────────────────────────┐
         │                              │                             │
         ▼                              ▼                             ▼
     Images/                    labels.csv                 metadata.json

   image.png              steering,throttle,          speed,objects,
                          brake                       lane,depth,etc.

                                        │
                                        ▼
                             YOUR CUSTOM DATASET
=========================================================================================
                                        │
                                        ▼
                             BEHAVIOUR CLONING TRAINING
=========================================================================================

               Images + Labels + Metadata
                         │
                         ▼
                    DataLoader
                         │
                         ▼
          Behaviour Cloning Network (PyTorch)
                         │
                         ▼
                 Behaviour Cloning Model.pt
                         │
                         ▼
               Can Already Drive Vehicle

=========================================================================================
                         │
                         ▼
                 REINFORCEMENT LEARNING
=========================================================================================

          Behaviour Cloning Weights
                     │
                     ▼
             Initialize PPO Policy
                     │
                     ▼
                MetaDrive Again
                     │
                     ▼
             AI Drives (NOT Expert)
                     │
                     ▼
        Reward Function Calculates Reward

        + Stay in lane
        + Reach waypoint
        + Smooth steering
        - Collision
        - Off road
        - Hard brake

                     │
                     ▼
             PPO Updates Policy
                     │
                     ▼
          Final Autonomous Driver.pt

=========================================================================================
                         │
                         ▼
                     INFERENCE
=========================================================================================

                      MetaDrive
                           │
                    RGB Camera
                           │
             Load Final PPO Model
                           │
                           ▼
              Steering / Brake / Throttle
                           │
                           ▼
                  MetaDrive env.step(action)
                           │
                           ▼
                    Vehicle Moves

                           │
────────────────────────────────────────────────────────────────────────────

        (PARALLEL PERCEPTION PIPELINE DURING INFERENCE)

                    RGB Camera
                         │
      ┌──────────────────┼─────────────────────┐
      ▼                  ▼                     ▼
    YOLO11          LaneNet                MiDaS
      │                  │                     │
      └──────────────────┴─────────────────────┘
                         │
                         ▼
                Scene Understanding
                         │
                         ▼
      Dashboard + Safety + Decision Explanation