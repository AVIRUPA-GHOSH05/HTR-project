from __future__ import division
from __future__ import print_function
import codecs
import os, sys

import tensorflow as tf

from DataLoader import FilePaths
rel_path = os.path.dirname(__file__)

class DecoderType:
    BestPath = 0
    WordBeamSearch = 1
    BeamSearch = 2


class Model:
    # Model Constants
    batchSize = 10 # 50
    imgSize = (800, 32)
    maxTextLen = 100

    def __init__(self, charList, decoderType=DecoderType.BestPath, mustRestore=False):
        self.charList = charList
        self.decoderType = decoderType
        self.mustRestore = mustRestore
        self.snapID = 0


        tf.compat.v1.disable_eager_execution()
        self.isTrain = tf.compat.v1.placeholder(tf.bool, name='is_train')
        self.inputImgs = tf.compat.v1.placeholder(
            tf.float32, shape=(None, None, Model.imgSize[1]))

        # setup CNN, RNN and CTC
        self.setupCNN()
        self.setupRNN()
        self.setupCTC()

        self.batchesTrained = 0
        updateOps = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(updateOps):
            self.optimizer = tf.compat.v1.train.AdamOptimizer().minimize(self.loss)

        # Initialize TensorFlow
        (self.sess, self.saver) = self.setupTF()

        self.training_loss_summary = tf.compat.v1.summary.scalar('loss', self.loss)
        self.writer = tf.compat.v1.summary.FileWriter(
           './logs', self.sess.graph)  # Tensorboard: Create writer
        self.merge = tf.compat.v1.summary.merge([self.training_loss_summary])  # Tensorboard: Merge

    def setupCNN(self):
        """Create the CNN layers used by the pretrained checkpoint."""
        cnnIn4d = tf.expand_dims(input=self.inputImgs, axis=3)
        kernelVals = [5, 5, 3, 3, 3]
        featureVals = [1, 32, 64, 128, 128, 256]
        poolVals = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2)]

        pool = cnnIn4d
        for i in range(len(poolVals)):
            kernel = tf.Variable(tf.random.truncated_normal(
                [kernelVals[i], kernelVals[i], featureVals[i], featureVals[i + 1]],
                stddev=0.1))
            conv = tf.nn.conv2d(
                input=pool, filters=kernel, padding='SAME', strides=(1, 1, 1, 1))
            convNorm = tf.compat.v1.layers.batch_normalization(
                conv, training=self.isTrain)
            relu = tf.nn.relu(convNorm)
            pool = tf.nn.max_pool2d(
                input=relu,
                ksize=(1, poolVals[i][0], poolVals[i][1], 1),
                strides=(1, poolVals[i][0], poolVals[i][1], 1),
                padding='VALID')

        self.cnnOut4d = pool

    def setupRNN(self):
        """Create the RNN layers used by the pretrained checkpoint."""
        rnnIn3d = tf.squeeze(self.cnnOut4d, axis=[2])
        numHidden = 256
        cells = [tf.compat.v1.nn.rnn_cell.LSTMCell(
            num_units=numHidden, state_is_tuple=True) for _ in range(2)]
        stacked = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        (forwardBackward, _) = tf.compat.v1.nn.bidirectional_dynamic_rnn(
            cell_fw=stacked, cell_bw=stacked, inputs=rnnIn3d, dtype=rnnIn3d.dtype)
        forward, backward = forwardBackward
        concat = tf.expand_dims(tf.concat([forward, backward], 2), 2)
        kernel = tf.Variable(tf.random.truncated_normal(
            [1, 1, numHidden * 2, len(self.charList) + 1], stddev=0.1))
        self.rnnOut3d = tf.squeeze(
            tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'),
            axis=[2])

    def setupCTC(self):
        """ Create CTC loss and decoder and return them """
        # BxTxC -> TxBxC
        self.ctcIn3dTBC = tf.transpose(self.rnnOut3d, [1, 0, 2])
        print("CTC in shape: ", self.ctcIn3dTBC.shape)

        # Ground truth text as sparse tensor
        with tf.name_scope('CTC_Loss'):
            self.gtTexts = tf.SparseTensor(tf.compat.v1.placeholder(tf.int64, shape=[
                                           None, 2]), tf.compat.v1.placeholder(tf.int32, [None]), tf.compat.v1.placeholder(tf.int64, [2]))
            # Calculate loss for batch
            self.seqLen = tf.compat.v1.placeholder(tf.int32, [None])
            self.loss = tf.reduce_mean(tf.compat.v1.nn.ctc_loss(labels=self.gtTexts, inputs=self.ctcIn3dTBC, sequence_length=self.seqLen,
                               ctc_merge_repeated=True, ignore_longer_outputs_than_inputs=True))
        with tf.name_scope('CTC_Decoder'):
            # Decoder: Best path decoding or Word beam search decoding
            if self.decoderType == DecoderType.BestPath:
                self.decoder = tf.nn.ctc_greedy_decoder(
                    inputs=self.ctcIn3dTBC, sequence_length=self.seqLen)
            elif self.decoderType == DecoderType.BeamSearch:
                self.decoder = tf.nn.ctc_beam_search_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, beam_width=50, merge_repeated=True)
            elif self.decoderType == DecoderType.WordBeamSearch:
                # Import compiled word beam search operation (see https://github.com/githubharald/CTCWordBeamSearch)
                word_beam_search_module = tf.load_op_library(
                    './TFWordBeamSearch.so')

                # Prepare: dictionary, characters in dataset, characters forming words
                chars = codecs.open(FilePaths.wordCharList.txt, 'r').read()
                wordChars = codecs.open(
                    FilePaths.fnWordCharList, 'r').read()
                corpus = codecs.open(FilePaths.corpus.txt, 'r').read()

                # # Decoder using the "NGramsForecastAndSample": restrict number of (possible) next words to at most 20 words: O(W) mode of word beam search
                # decoder = word_beam_search_module.word_beam_search(tf.nn.softmax(ctcIn3dTBC, dim=2), 25, 'NGramsForecastAndSample', 0.0, corpus.encode('utf8'), chars.encode('utf8'), wordChars.encode('utf8'))

                # Decoder using the "Words": only use dictionary, no scoring: O(1) mode of word beam search
                self.decoder = word_beam_search_module.word_beam_search(tf.nn.softmax(
                    self.ctcIn3dTBC, dim=2), 25, 'Words', 0.0, corpus.encode('utf8'), chars.encode('utf8'), wordChars.encode('utf8'))

        # Return a CTC operation to compute the loss and CTC operation to decode the RNN output
        return self.loss, self.decoder

    def setupTF(self):
        """ Initialize TensorFlow """
        print('Python: ' + sys.version)
        print('Tensorflow: ' + tf.__version__)
        sess = tf.compat.v1.Session()  # Tensorflow session
        saver = tf.compat.v1.train.Saver(max_to_keep=3)  # Saver saves model to file
        modelDir = rel_path + '/' + '../pretrained/'
        latestSnapshot = tf.train.latest_checkpoint(modelDir)  # Is there a saved model?
        # If model must be restored (for inference), there must be a snapshot
        if self.mustRestore and not latestSnapshot:
            raise Exception('No saved model found in: ' + modelDir)
        # Load saved model if available
        if latestSnapshot:
            print('Init with stored values from ' + latestSnapshot)
            saver.restore(sess, latestSnapshot)
        else:
            print('Init with new values')
            sess.run(tf.compat.v1.global_variables_initializer())

        return (sess, saver)

    def toSpare(self, texts):
        """ Convert ground truth texts into sparse tensor for ctc_loss """
        indices = []
        values = []
        shape = [len(texts), 0]  # Last entry must be max(labelList[i])
        # Go over all texts
        for (batchElement, texts) in enumerate(texts):
            # Convert to string of label (i.e. class-ids)
            # print(texts)
            # labelStr = []
            # for c in texts:
            #     print(c, '|', end='')
            #     labelStr.append(self.charList.index(c))
            # print(' ')
            labelStr = [self.charList.index(c) for c in texts]
            # Sparse tensor must have size of max. label-string
            if len(labelStr) > shape[1]:
                shape[1] = len(labelStr)
            # Put each label into sparse tensor
            for (i, label) in enumerate(labelStr):
                indices.append([batchElement, i])
                values.append(label)

        return (indices, values, shape)

    def decoderOutputToText(self, ctcOutput, batchSize):
        """ Extract texts from output of CTC decoder """
        # Contains string of labels for each batch element
        encodedLabelStrs = [[] for i in range(batchSize)]
        # Word beam search: label strings terminated by blank
        if self.decoderType == DecoderType.WordBeamSearch:
            blank = len(self.charList)
            for b in range(batchSize):
                for label in ctcOutput[b]:
                    if label == blank:
                        break
                    encodedLabelStrs[b].append(label)
        # TF decoders: label strings are contained in sparse tensor
        else:
            # Ctc returns tuple, first element is SparseTensor
            decoded = ctcOutput[0][0]
            for (idx, idx2d) in enumerate(decoded.indices):
                label = decoded.values[idx]
                batchElement = idx2d[0]  # index according to [b,t]
                encodedLabelStrs[batchElement].append(label)
        # Map labels to chars for all batch elements
        return [str().join([self.charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]

    def trainBatch(self, batch, batchNum):
        """ Feed a batch into the NN to train it """
        sparse = self.toSpare(batch.gtTexts)
        evalList = [self.merge, self.optimizer, self.loss]
        batchSize = len(batch.imgs)
        sequenceLength = batch.imgs[0].shape[0] // 4
        feedDict = {
            self.inputImgs: batch.imgs,
            self.gtTexts: sparse,
            self.seqLen: [sequenceLength] * batchSize,
            self.isTrain: True,
        }
        (loss_summary, _, lossVal) = self.sess.run(evalList, feedDict)
        # Tensorboard: Add loss_summary to writer
        self.writer.add_summary(loss_summary, batchNum)
        self.batchesTrained += 1
        return lossVal

    def return_rnn_out(self, batch, write_on_csv=False):
        """Only return rnn_out prediction value without decoded"""
        numBatchElements = len(batch.imgs)
        sequenceLength = batch.imgs[0].shape[0] // 4
        rnnOutput = self.sess.run(
            self.ctcIn3dTBC,
            {
                self.inputImgs: batch.imgs,
                self.seqLen: [sequenceLength] * numBatchElements,
                self.isTrain: False,
            })

        if write_on_csv:
            s = rnnOutput.shape
            b = 0
            csv = ''
            for t in range(s[0]):
                for c in range(s[2]):
                    csv += str(rnnOutput[t, b, c]) + ';'
                csv += '\n'
            open('mat_0.csv', 'w').write(csv)

        return rnnOutput[:, 0, :]

    def inferBatch(self, batch):
        """ Feed a batch into the NN to recognize texts """
        numBatchElements = len(batch.imgs)
        sequenceLength = batch.imgs[0].shape[0] // 4
        feedDict = {
            self.inputImgs: batch.imgs,
            self.seqLen: [sequenceLength] * numBatchElements,
            self.isTrain: False,
        }
        evalRes = self.sess.run([self.decoder, self.ctcIn3dTBC], feedDict)
        decoded = evalRes[0]
        # # Dump RNN output to .csv file
        # decoded, rnnOutput = self.sess.run([self.decoder, self.rnnOutput], {
        #                                    self.inputImgs: batch.imgs, self.seqLen: [Model.maxTextLen] * Model.batchSize})
        # s = rnnOutput.shape
        # b = 0
        # csv = ''
        # for t in range(s[0]):
        #     for c in range(s[2]):
        #         csv += str(rnnOutput[t, b, c]) + ';'
        #     csv += '\n'
        # open('mat_0.csv', 'w').write(csv)

        texts = self.decoderOutputToText(decoded, numBatchElements)
        return texts

    def save(self):
        """ Save model to file """
        self.snapID += 1
        self.saver.save(self.sess, '../model/snapshot',
                        global_step=self.snapID)
