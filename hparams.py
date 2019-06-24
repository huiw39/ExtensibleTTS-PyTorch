class hparams:

    fs = 16000
    frame_period = 5
    hop_length = 80
    fftlen = 1024
    alpha = 0.41

    mgc_dim = 180
    lf0_dim = 3
    vuv_dim = 1
    bap_dim = 3
    mgc_start_idx = 0
    lf0_start_idx = 180
    vuv_start_idx = 183
    bap_start_idx = 184
    duration_linguistic_dim = 416
    acoustic_linguistic_dim = 425
    duration_dim = 5
    acoustic_dim = mgc_dim + lf0_dim + vuv_dim + bap_dim

    # training parameters
    batch_size = 16
    hidden_size = 256
    num_layers = 2
    epochs = 200
    checkpoint_interval = 1000  # Steps between writing checkpoints

    accumulation_steps = 8
    grad_norm = 10
    # learning rate parameters
    init_learning_rate = 1e-3
    lr_schedule_type = 'step'  # or 'noam'
    # for noam learning rate schedule
    noam_warm_up_steps = 2000 * (batch_size // 16)
    # for step learning rate schedule
    step_gamma = 0.5
    lr_step_interval = 15000

    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_eps = 1e-8
    amsgrad = False
    weight_decay = 0.0
    # modify if one wants to use a fixed learning rate, else set to None to use noam learning rate
    fixed_learning_rate = None
    # -----------------
