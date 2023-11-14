HYPERPARAMETERS = {
    'bert': {
        'epochs': 5,
    },

    'lstm': {
        'lstm_vocab_size': 70,
        'lstm_window_size': 10,
        'lstm_embedding_dim': 2,
        'lstm_hidden_dim': 16,
        'lstm_dense_dim': 32,
        'lstm_n_layers': 1,
        'lstm_max_norm': 2,
    },

    'cnn': {
        'cnn_vocab_size': 70,
        'cnn_embedding_dim': 2,
        'cnn_num_filters': 100,
        'cnn_filter_sizes':[3, 4, 5],
        'cnn_output_dim':1,  # Binary classification
        'cnn_dropout':0.5
    },
}