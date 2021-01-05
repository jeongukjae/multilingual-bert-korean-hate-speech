import csv

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub
import tensorflow_text as text

gender_bias_label_to_index = {'False': 0, 'True': 1}
bias_label_to_index = {'none': 0, 'others': 1, 'gender': 2}
hate_label_to_index = {'none': 0, 'offensive': 1, 'hate': 2}

def read_dataset(key):
    with open(f'./korean-hate-speech/labeled/{key}.tsv') as comment, open(f'./korean-hate-speech/news_title/{key}.news_title.txt') as title:
        next(comment)

        title_list = [line for line in csv.reader(title)]
        comment_list = [line for line in csv.reader(comment, delimiter='\t')]

        dataset = [
            [tf.constant(title[0]), tf.constant(comment[0])]
            for title, comment in zip(title_list, comment_list)
        ]
        labels = [
            (
                tf.one_hot(tf.constant(gender_bias_label_to_index[comment[1]]), 2),
                tf.one_hot(tf.constant(bias_label_to_index[comment[2]]), 3),
                tf.one_hot(tf.constant(hate_label_to_index[comment[3]]), 3),
            )
            for comment in comment_list
        ]

    return dataset, labels

train_dataset, train_labels = read_dataset("train")
dev_dataset, dev_labels = read_dataset("dev")

with open('./korean-hate-speech/test.no_label.tsv') as comment, open('./korean-hate-speech/news_title/test.news_title.txt') as title:
    next(comment)

    test_comment = [line for line in csv.reader(comment, delimiter='\t')]
    test_title = [line for line in csv.reader(title)]

    test_dataset = [
        [tf.constant(title[0]), tf.constant(comment[0])]
        for title, comment in zip(test_title, test_comment)
    ]

print(len(train_dataset), len(dev_dataset), len(test_dataset))

train_set = (
    tf.data.Dataset.from_tensor_slices((
        train_dataset,
        [label[0] for label in train_labels],
        [label[1] for label in train_labels],
        [label[2] for label in train_labels],
    ))
    .shuffle(len(train_labels), reshuffle_each_iteration=True)
    .batch(32)
    .map(lambda *x: ((x[0][:,0], x[0][:,1]), (x[1], x[2], x[3])))
    .repeat()
)
dev_set = (
    tf.data.Dataset.from_tensor_slices((
        dev_dataset,
        [label[0] for label in dev_labels],
        [label[1] for label in dev_labels],
        [label[2] for label in dev_labels],
    ))
    .batch(64)
    .map(lambda *x: ((x[0][:,0], x[0][:,1]), (x[1], x[2], x[3])))
)
test_set = tf.data.Dataset.from_tensor_slices(test_dataset).map(lambda x: [x[0], x[1]])

#
# 모델 선언부
#
preprocessor = hub.load("https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/2")
tokenize = hub.KerasLayer(preprocessor.tokenize)
bert_pack_inputs = hub.KerasLayer(preprocessor.bert_pack_inputs, arguments=dict(seq_length=128))
encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3", trainable=True)

def create_model():
    # [context, comment]
    context_inputs = tf.keras.layers.Input(shape=(), dtype=tf.string)
    comment_inputs = tf.keras.layers.Input(shape=(), dtype=tf.string)

    preprocessed = bert_pack_inputs([tokenize(context_inputs), tokenize(comment_inputs)])

    # pooling only segments 2
    encoder_outputs = encoder(preprocessed)["pooled_output"]
    gender_prob = tf.keras.layers.Dense(2, activation='softmax', name='gender')(encoder_outputs)
    bias_prob = tf.keras.layers.Dense(3, activation='softmax', name='bias')(encoder_outputs)
    hate_prob = tf.keras.layers.Dense(3, activation='softmax', name='hate')(encoder_outputs)

    model = tf.keras.Model([context_inputs, comment_inputs], [gender_prob, bias_prob, hate_prob])
    return model

model = create_model()
model.compile(
    optimizer=tf.keras.optimizers.Adamax(learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=4e-5,
        decay_steps=10,
        decay_rate=0.95,
    )),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[
        ['acc', tfa.metrics.F1Score(num_classes=2, average='macro', name='f1')],
        ['acc', tfa.metrics.F1Score(num_classes=3, average='macro', name='f1')],
        ['acc', tfa.metrics.F1Score(num_classes=3, average='macro', name='f1')],
    ]
)
model.fit(train_set, epochs=25, validation_data=dev_set, steps_per_epoch=50)

# kaggle 확인용
gender_result = []
bias_result = []
hate_result = []
for item in test_set.batch(32):
    result = model.predict_on_batch(item)

    gender_result.append(result[0])
    bias_result.append(result[1])
    hate_result.append(result[2])

gender_result = tf.concat(gender_result, axis=0)
bias_result = tf.concat(bias_result, axis=0)
hate_result = tf.concat(hate_result, axis=0)

gender_predicted = tf.argmax(gender_result, axis=-1)
bias_predicted = tf.argmax(bias_result, axis=-1)
hate_predicted = tf.argmax(hate_result, axis=-1)

print(tf.size(gender_predicted))
print(tf.size(bias_predicted))
print(tf.size(hate_predicted))

def save_submission_result(key, labels):
    with open('./korean-hate-speech/test.no_label.tsv') as test_file:
        reader = csv.DictReader(test_file, delimiter='\t')
        lines = [line for line in reader]

    print(len(lines), lines[0])

    with open(f'./{key}_prediced.csv', 'w') as result_file:
        writer = csv.DictWriter(result_file, fieldnames=['comments', 'label'])
        writer.writeheader()

        for line, label in zip(lines, labels.numpy().tolist()):
            writer.writerow({'comments': line['comments'], 'label': label})

save_submission_result("gender", gender_predicted)
save_submission_result("bias", bias_predicted)
save_submission_result("hate", hate_predicted)

# saved model
final_model = create_model()
final_model.set_weights(model.get_weights())

@tf.function
def _serve_model(context, comment):
    return final_model([context, comment])

tf.saved_model.save(
    final_model,
    "./hate-speech-model/0",
    signatures=_serve_model.get_concrete_function(
        tf.TensorSpec(shape=[None], dtype=tf.string, name='context'),
        tf.TensorSpec(shape=[None], dtype=tf.string, name='comment')
    )
)

# 그냥 같은 모델인지 확인용
print(model([tf.constant(["뉴스기사-뉴스기사"]), tf.constant(["ㅋㅋㅋㅋ 그래도 조아해주는 팬들 많아서 좋겠다 ㅠㅠ 니들은 온유가 안만져줌 ㅠㅠ"])]))
print(final_model([tf.constant(["뉴스기사-뉴스기사"]), tf.constant(["ㅋㅋㅋㅋ 그래도 조아해주는 팬들 많아서 좋겠다 ㅠㅠ 니들은 온유가 안만져줌 ㅠㅠ"])]))
