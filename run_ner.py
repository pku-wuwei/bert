# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT for named entity recognition."""

import csv
import os
from typing import List

import tensorflow as tf
from tensorflow.contrib import crf

import modeling
import optimization
import tokenization

flags = tf.flags

FLAGS = flags.FLAGS

# Required parameters
flags.DEFINE_string(
    "data_dir", None, "The input data dir. Should contain the .tsv files (or other data files) for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None, "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("output_dir", None, "The output directory where the model checkpoints will be written.")

# Other parameters

flags.DEFINE_string("init_checkpoint", None, "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", False, "Whether to run the model in inference mode on the test set.")

flags.DEFINE_bool("use_crf", False, "Whether to use the crf after the BERT encoder.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 1, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0, "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 200, "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 200, "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer("num_tpu_cores", 8, "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid: List[int], text: List[str], label: List[str] = None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example. [instance_id, label_id]
          text: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids: List[int],
                 input_mask: List[int],
                 segment_ids: List[int],
                 label_mask: List[int],
                 label_ids: List[int],
                 guid: List[int],
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_mask = label_mask
        self.label_ids = label_ids
        self.guid = guid
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    data_types = ['train', 'dev', 'test']

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class MSRAProcessor(DataProcessor):
    """Processor for the MSRA data set."""

    def __init__(self):
        self.language = "zh"

    def get_train_examples(self, data_dir):
        """See base class."""
        return self.get_examples(data_dir, data_type='train')

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self.get_examples(data_dir, data_type='dev')

    def get_test_examples(self, data_dir):
        """See base class."""
        return self.get_examples(data_dir, data_type='test')

    def get_examples(self, data_dir, data_type):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, data_type + ".tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            guid = [self.data_types.index(data_type), i]
            text = tokenization.convert_to_unicode(line[0]).split()
            label = tokenization.convert_to_unicode(line[1]).split()
            assert len(text) == len(label), "text length {} != label length {}".format(len(text), len(label))
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples

    def get_labels(self):
        """See base class."""
        return ['O', 'B-NR', 'B-NS', 'B-NT', 'E-NR', 'E-NS', 'E-NT', 'M-NR', 'M-NS', 'M-NT', 'S-NR', 'S-NS', 'S-NT']


def convert_single_example(example: InputExample,
                           label_list: List[str],
                           max_seq_length: int,
                           tokenizer: tokenization.FullTokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_mask=[0] * max_seq_length,
            label_ids=[0] * max_seq_length,
            guid=[0, 0],
            is_real_example=False,
        )
    tokens = ['[CLS]']
    labels = ['O']
    label_mask = [0]

    for word, label in zip(example.text, example.label):
        word_tokens = tokenizer.tokenize(word)
        tokens.extend(word_tokens)
        for i, token in enumerate(word_tokens):
            if i == 0:
                labels.append(label)
                label_mask.append(1)
            else:
                labels.append('O')
                label_mask.append(0)
    tokens = tokens[: max_seq_length - 1] + ['[SEP]']
    labels = labels[: max_seq_length - 1] + ['O']
    label_mask = label_mask[: max_seq_length - 1] + [0]

    input_ids = tokenizer.convert_tokens_to_ids(tokens) + [0] * (max_seq_length - len(tokens))
    input_mask = [1] * len(tokens) + [0] * (max_seq_length - len(tokens))
    segment_ids = [0] * max_seq_length
    label_ids = [label_list.index(l) for l in labels] + [0] * (max_seq_length - len(tokens))
    label_mask = label_mask + [0] * (max_seq_length - len(tokens))

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(label_mask) == max_seq_length

    if example.guid[1] < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % example.guid)
        tf.logging.info("tokens: %s" % tokens)
        tf.logging.info("labels: %s" % labels)

        tf.logging.info("input_ids: %s" % input_ids)
        tf.logging.info("input_mask: %s" % input_mask)
        tf.logging.info("segment_ids: %s" % segment_ids)
        tf.logging.info("label_ids: %s" % label_ids)
        tf.logging.info("label_mask: %s" % label_mask)

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        label_mask=label_mask,
        is_real_example=True,
        guid=example.guid,
    )
    return feature


def file_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""
    if tf.gfile.Exists(output_file):
        print(output_file, " 已存在，不再创建")
        return
    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(example, label_list, max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = {"input_ids": create_int_feature(feature.input_ids),
                    "input_mask": create_int_feature(feature.input_mask),
                    "segment_ids": create_int_feature(feature.segment_ids),
                    "label_ids": create_int_feature(feature.label_ids),
                    "label_mask": create_int_feature(feature.label_mask),
                    "guid": create_int_feature(feature.guid),
                    "is_real_example": create_int_feature([int(feature.is_real_example)])}

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
        "guid": tf.FixedLenFeature([2], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32. So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def batch_boolean_mask(batch_input, batch_mask, input_rank):
    """
    每个batch取出所需的tensor
    :param batch_input: (batch, seq_len, hidden)
    :param batch_mask: (batch, seq_len)
    :return: (batch, seq_len, hidden)
    """
    outputs = []
    for input_tensor, mask in zip(tf.unstack(batch_input), tf.unstack(batch_mask)):
        output_tensor = tf.boolean_mask(input_tensor, mask)
        paddings = [[0, tf.shape(batch_input)[1]-tf.shape(output_tensor)[0]]]
        if input_rank == 3:
            paddings.extend([[0, 0]])
        # paddings = tf.Print(paddings, [output_tensor, paddings])
        padded_output = tf.pad(output_tensor, paddings, 'CONSTANT', constant_values=-1)  # (seq, dim)
        outputs.append(padded_output)
    output_tensors = tf.stack(outputs, axis=0)
    return output_tensors


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 label_ids, label_mask, num_labels, use_one_hot_embeddings, use_crf=False):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    output_layer = model.get_sequence_output()  # (batch_size, sequence_len, hidden_size)
    valid_output = batch_boolean_mask(output_layer, label_mask, 3)  # (batch_size, sequence_len, hidden_size)
    valid_labels = batch_boolean_mask(label_ids, label_mask, 2)  # (batch_size, sequence_len)
    hidden_size = output_layer.shape[-1].value
    sequence_length = tf.reduce_sum(label_mask, axis=-1)  # (batch_size)
    sequence_mask = tf.to_float(tf.sequence_mask(sequence_length, tf.shape(label_mask)[1]))
    with tf.variable_scope("loss"):
        valid_output = tf.nn.dropout(valid_output, keep_prob=0.9 if is_training else 1)
        middle_logits = tf.layers.dense(valid_output, units=hidden_size, activation=tf.tanh)
        logits = tf.layers.dense(middle_logits, units=num_labels, activation=tf.tanh)  # (batch, seq_len, num_labels)
        if use_crf:
            trans = tf.get_variable("transitions", shape=[num_labels, num_labels])
            log_likelihood, trans = crf.crf_log_likelihood(
                inputs=logits,
                tag_indices=valid_labels,
                transition_params=trans,
                sequence_lengths=sequence_length)
            pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=sequence_length)
            return tf.reduce_mean(-log_likelihood), -log_likelihood, pred_ids
        else:
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            one_hot_labels = tf.one_hot(valid_labels, depth=num_labels, dtype=tf.float32)
            per_token_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)  # (batch, seq_len)
            per_example_loss = tf.reduce_mean(per_token_loss * sequence_mask, -1)
            loss = tf.reduce_mean(per_example_loss)
            pred_ids = tf.argmax(tf.nn.softmax(logits, axis=-1), -1)
            return loss, per_example_loss, pred_ids


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu, use_one_hot_embeddings, use_crf=False):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        label_mask = features["label_mask"]
        is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        total_loss, per_example_loss, pred_ids = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids, label_mask,
            num_labels, use_one_hot_embeddings, use_crf)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ", *INIT_FROM_CKPT*" if var.name in initialized_variable_names else ""
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            logging_hook = tf.train.LoggingTensorHook({"loss": total_loss}, every_n_iter=40)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                training_hooks=[logging_hook],
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(label_ids, label_mask, pred_ids):
                return {
                    "eval_loss": tf.metrics.mean_squared_error(labels=label_ids, predictions=pred_ids, weights=label_mask),
                }
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=(metric_fn, [label_ids, label_mask, pred_ids]),
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"pred_ids": pred_ids, "guid": features['guid']},
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "msra": MSRAProcessor,
    }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case, FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError("At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError("Cannot use sequence length %d because the BERT model was only trained up to sequence length "
                         "%d" % (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)

    processor = processors[task_name]()

    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        use_crf=FLAGS.use_crf
    )

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(train_examples, label_list, FLAGS.max_seq_length,
                                                tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples = len(eval_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % FLAGS.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            assert len(eval_examples) % FLAGS.eval_batch_size == 0
            eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(predict_examples) % FLAGS.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, label_list, FLAGS.max_seq_length, tokenizer,
                                                predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            tf.logging.info("***** Predict results *****")
            for (i, prediction) in enumerate(result):
                output_line = " ".join(str(guid) for guid in prediction["guid"]) + "\t"
                if i >= num_actual_predict_examples:
                    break
                output_line += "\t".join(
                    str(class_probability) for class_probability in prediction["probabilities"]) + "\n"
                writer.write(output_line)
                num_written_lines += 1
        assert num_written_lines == num_actual_predict_examples


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
