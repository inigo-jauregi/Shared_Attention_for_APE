from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import onmt
from onmt.Utils import aeq
from onmt.Utils import pack_padded_not_sorted_sequence as pack_not_sorted
from onmt.Utils import pad_packed_not_sorted_sequence as unpack_not_sorted


class EncoderBase(nn.Module):
    """
    EncoderBase class for sharing code among various encoder.
    """
    def _check_args(self, input, lengths=None, hidden=None):
        s_len, n_batch, n_feats = input.size()
        if lengths is not None:
            n_batch_, = lengths.size()
            aeq(n_batch, n_batch_)

    def _check_args_double_enc(self, input, lengths_src=None, lengths_inter=None, hidden_src=None, hidden_inter=None):
        #SRC
        s_len, n_batch, n_feats = input[0].size()
        if lengths_src is not None:
            n_batch_, = lengths_src.size()
            aeq(n_batch, n_batch_)
        #INTER
        s_len, n_batch, n_feats = input[1].size()
        if lengths_inter is not None:
            n_batch_, = lengths_inter.size()
            aeq(n_batch, n_batch_)

    def forward(self, input, lengths=None, hidden=None):
        """
        Args:
            input (LongTensor): len x batch x nfeat.
            lengths (LongTensor): batch
            hidden: Initial hidden state.
        Returns:
            hidden_t (Variable): Pair of layers x batch x rnn_size - final
                                    encoder state
            outputs (FloatTensor):  len x batch x rnn_size -  Memory bank
        """
        raise NotImplementedError


class MeanEncoder(EncoderBase):
    """ A trivial encoder without RNN, just takes mean as final state. """
    def __init__(self, num_layers, embeddings):
        super(MeanEncoder, self).__init__()
        self.num_layers = num_layers
        self.embeddings = embeddings

    def forward(self, input, lengths=None, hidden=None):
        """ See EncoderBase.forward() for description of args and returns. """
        self._check_args(input, lengths, hidden)

        emb = self.embeddings(input)
        s_len, batch, emb_dim = emb.size()
        mean = emb.mean(0).expand(self.num_layers, batch, emb_dim)
        return (mean, mean), emb


class RNNEncoder(EncoderBase):
    """ The standard RNN encoder. """
    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout, embeddings):
        super(RNNEncoder, self).__init__()

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings
        self.no_pack_padded_seq = False

        # Use pytorch version when available.
        if rnn_type == "SRU":
            # SRU doesn't support PackedSequence.
            self.no_pack_padded_seq = True
            self.rnn = onmt.modules.SRU(
                    input_size=embeddings.embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    bidirectional=bidirectional)
        else:
            self.rnn = getattr(nn, rnn_type)(
                    input_size=embeddings.embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    bidirectional=bidirectional)

    def forward(self, input, lengths=None, hidden=None):
        """ See EncoderBase.forward() for description of args and returns."""
        self._check_args(input, lengths, hidden)

        emb = self.embeddings(input)
        s_len, batch, emb_dim = emb.size()

        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Variable.
            lengths = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths)

        outputs, hidden_t = self.rnn(packed_emb, hidden)

        if lengths is not None and not self.no_pack_padded_seq:
            outputs = unpack(outputs)[0]

        return hidden_t, outputs

class DoubleRNNEncoder(EncoderBase):
    """ The standard RNN encoder. """
    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout, embeddings_src, embeddings_inter):
        super(DoubleRNNEncoder, self).__init__()

        num_directions = 2 if bidirectional else 1
        if bidirectional == True:
            print ('Bidirectional RNN!')
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings_src = embeddings_src
        self.embeddings_inter = embeddings_inter
        self.no_pack_padded_seq = False

        # Use pytorch version when available.
        if rnn_type == "SRU":
            # SRU doesn't support PackedSequence.
            self.no_pack_padded_seq = True
            self.rnn1 = onmt.modules.SRU(
                    input_size=embeddings_src.embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    bidirectional=bidirectional)
            self.rnn2 = onmt.modules.SRU(
                input_size=embeddings_inter.embedding_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional)
        else:
            self.rnn1 = getattr(nn, rnn_type)(
                    input_size=embeddings_src.embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    bidirectional=bidirectional)
            self.rnn2 = getattr(nn, rnn_type)(
                input_size=embeddings_inter.embedding_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional)

    def forward(self, input, lengths_src=None, lengths_inter=None, hidden_src=None, hidden_inter=None):
        """ See EncoderBase.forward() for description of args and returns."""
        self._check_args_double_enc(input, lengths_src, lengths_inter,hidden_src, hidden_inter)

        #First source
        src_emb = self.embeddings_src(input[0])
        src_s_len, src_batch, src_emb_dim = src_emb.size()

        packed_src_emb = src_emb
        if lengths_src is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Variable.
            lengths_src = lengths_src.view(-1).tolist()
            packed_src_emb = pack(src_emb, lengths_src)

        src_outputs, src_hidden_t = self.rnn1(packed_src_emb, hidden_src)

        if lengths_src is not None and not self.no_pack_padded_seq:
            src_outputs = unpack(src_outputs)[0]

        #Second inter
        inter_emb = self.embeddings_inter(input[1])
        inter_s_len, inter_batch, inter_emb_dim = inter_emb.size()

        packed_inter_emb = inter_emb
        if lengths_inter is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Variable.
            lengths_inter = lengths_inter.view(-1).tolist()
            #packed_inter_emb = pack_not_sorted(inter_emb, lengths_inter)

        inter_outputs, inter_hidden_t = self.rnn2(inter_emb, hidden_inter) #WHAT IS THE DIFFERENCE BETWEEN THEM???
                                                                           #DOES THE OUTPUT GO THROUGH A SOFTMAX???

        #if lengths_inter is not None and not self.no_pack_padded_seq:
        #    inter_outputs = unpack_not_sorted(inter_outputs)[0]

        return src_hidden_t, src_outputs, inter_hidden_t, inter_outputs


class RNNDecoderBase(nn.Module):
    """
    RNN decoder base class.
    """
    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type, coverage_attn, context_gate,
                 copy_attn, dropout, embeddings):
        super(RNNDecoderBase, self).__init__()

        # Basic attributes.
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)

        # Build the RNN.
        self.rnn = self._build_rnn(rnn_type, self._input_size, hidden_size,
                                   num_layers, dropout)

        # Set up the context gate.
        self.context_gate = None
        if context_gate is not None:
            self.context_gate = onmt.modules.ContextGateFactory(
                context_gate, self._input_size,
                hidden_size, hidden_size, hidden_size
            )

        # Set up the standard attention.
        self._coverage = coverage_attn
        self.attn = onmt.modules.GlobalAttention(
            hidden_size, coverage=coverage_attn,
            attn_type=attn_type
        )

        # Set up a separated copy attention layer, if needed.
        self._copy = False
        if copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(
                hidden_size, attn_type=attn_type
            )
            self._copy = True

    def forward(self, input, context, state, context_inter=None):
        """
        Forward through the decoder.
        Args:
            input (LongTensor): a sequence of input tokens tensors
                                of size (len x batch x nfeats).
            context (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
        Returns:
            outputs (FloatTensor): a Tensor sequence of output from the decoder
                                   of shape (len x batch x hidden_size).
            state (FloatTensor): final hidden state from the decoder.
            attns (dict of (str, FloatTensor)): a dictionary of different
                                type of attention Tensor from the decoder
                                of shape (src_len x batch).
        """
        # Args Check
        if context_inter:
            assert isinstance(state, RNNDecoderState_double_enc)
        else:
            assert isinstance(state,RNNDecoderState)
        input_len, input_batch, _ = input.size()
        contxt_len, contxt_batch, _ = context.size()
        if context_inter:
            contxt_inter_len, contxt_inter_batch,_=context_inter.size()
            aeq(input_batch, contxt_batch,contxt_inter_batch)
        else:
            aeq(input_batch,contxt_batch)
        # END Args Check

        # Run the forward pass of the RNN.
        hidden, outputs, attns, coverage = \
            self._run_forward_pass(input, context, state, context_inter)

        # Update the state with the result.
        final_output = outputs[-1]
        state.update_state(hidden, final_output.unsqueeze(0),
                           coverage.unsqueeze(0)
                           if coverage is not None else None)

        # Concatenates sequence of tensors along a new dimension.
        outputs = torch.stack(outputs)
        for k in attns:
            attns[k] = torch.stack(attns[k])

        return outputs, state, attns

    def _fix_enc_hidden(self, h):
        """
        The encoder hidden is  (layers*directions) x batch x dim.
        We need to convert it to layers x batch x (directions*dim).
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def init_decoder_state(self, src, context_src, enc_hidden_src, context_inter=None, enc_hidden_inter=None):
        if isinstance(enc_hidden_src, tuple):  # GRU
            if context_inter:
                return RNNDecoderState_double_enc(context_src, self.hidden_size,
                                                  tuple([self._fix_enc_hidden(enc_hidden_src[i])
                                                         for i in range(len(enc_hidden_src))]),
                                                  context_inter, tuple([self._fix_enc_hidden(enc_hidden_inter[i])
                                                                        for i in range(len(enc_hidden_inter))]))
            else:
                return RNNDecoderState(context_src, self.hidden_size,
                                       tuple([self._fix_enc_hidden(enc_hidden_src[i])
                                              for i in range(len(enc_hidden_src))]))
        else:  # LSTM
            return RNNDecoderState(context_src, self.hidden_size,
                                   self._fix_enc_hidden(enc_hidden_src))


class StdRNNDecoder(RNNDecoderBase):
    """
    Stardard RNN decoder, with Attention.
    Currently no 'coverage_attn' and 'copy_attn' support.
    """
    def _run_forward_pass(self, input, context, state,context_inter=None):
        """
        Private helper for running the specific RNN forward pass.
        Must be overriden by all subclasses.
        Args:
            input (LongTensor): a sequence of input tokens tensors
                                of size (len x batch x nfeats).
            context (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
        Returns:
            hidden (Variable): final hidden state from the decoder.
            outputs ([FloatTensor]): an array of output of every time
                                     step from the decoder.
            attns (dict of (str, [FloatTensor]): a dictionary of different
                            type of attention Tensor array of every time
                            step from the decoder.
            coverage (FloatTensor, optional): coverage from the decoder.
        """
        assert not self._copy  # TODO, no support yet.
        assert not self._coverage  # TODO, no support yet.

        # Initialize local and return variables.
        outputs = []
        attns = {"std": []}
        coverage = None

        emb = self.embeddings(input)

        # Run the forward pass of the RNN.
        #Maybe we need to change something here!
        rnn_output, hidden = self.rnn(emb, state.hidden)
        # Result Check
        input_len, input_batch, _ = input.size()
        output_len, output_batch, _ = rnn_output.size()
        aeq(input_len, output_len)
        aeq(input_batch, output_batch)
        # END Result Check

        # Calculate the attention.
        attn_outputs, attn_scores = self.attn(
            rnn_output.transpose(0, 1).contiguous(),  # (output_len, batch, d)
            context.transpose(0, 1)                   # (contxt_len, batch, d)
        )
        attns["std"] = attn_scores

        # Calculate the context gate.
        if self.context_gate is not None:
            outputs = self.context_gate(
                emb.view(-1, emb.size(2)),
                rnn_output.view(-1, rnn_output.size(2)),
                attn_outputs.view(-1, attn_outputs.size(2))
            )
            outputs = outputs.view(input_len, input_batch, self.hidden_size)
            outputs = self.dropout(outputs)
        else:
            outputs = self.dropout(attn_outputs)    # (input_len, batch, d)

        # Return result.
        return hidden, outputs, attns, coverage

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        """
        Private helper for building standard decoder RNN.
        """
        # Use pytorch version when available.
        if rnn_type == "SRU":
            return onmt.modules.SRU(
                    input_size, hidden_size,
                    num_layers=num_layers,
                    dropout=dropout)

        return getattr(nn, rnn_type)(
            input_size, hidden_size,
            num_layers=num_layers,
            dropout=dropout)

    @property
    def _input_size(self):
        """
        Private helper returning the number of expected features.
        """
        return self.embeddings.embedding_size


class InputFeedRNNDecoder(RNNDecoderBase):
    """
    Stardard RNN decoder, with Input Feed and Attention.
    """
    def _run_forward_pass(self, input, context, state, conterxt_inter=None):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        # Additional args check.
        output = state.input_feed.squeeze(0)
        output_batch, _ = output.size()
        input_len, input_batch, _ = input.size()
        aeq(input_batch, output_batch)
        # END Additional args check.

        # Initialize local and return variables.
        outputs = []
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []

        emb = self.embeddings(input)
        assert emb.dim() == 3  # len x batch x embedding_dim

        hidden = state.hidden
        coverage = state.coverage.squeeze(0) \
            if state.coverage is not None else None

        # Input feed concatenates hidden state with
        # input at every time step.!!! WHAT DOES THIS MEAN???
        for i, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)
            emb_t = torch.cat([emb_t, output], 1)

            rnn_output, hidden = self.rnn(emb_t, hidden)
            attn_output, attn = self.attn(rnn_output,
                                          context.transpose(0, 1))
            if self.context_gate is not None:
                output = self.context_gate(
                    emb_t, rnn_output, attn_output
                )
                output = self.dropout(output)
            else:
                output = self.dropout(attn_output)
            outputs += [output]
            attns["std"] += [attn]

            # Update the coverage attention.
            if self._coverage:
                coverage = coverage + attn \
                    if coverage is not None else attn
                attns["coverage"] += [coverage]

            # Run the forward pass of the copy attention layer.
            if self._copy:
                _, copy_attn = self.copy_attn(output,
                                              context.transpose(0, 1))
                attns["copy"] += [copy_attn]

        # Return result.
        return hidden, outputs, attns, coverage

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert not rnn_type == "SRU", "SRU doesn't support input feed! " \
                "Please set -input_feed 0!"
        if rnn_type == "LSTM":
            stacked_cell = onmt.modules.StackedLSTM
        else:
            stacked_cell = onmt.modules.StackedGRU
        return stacked_cell(num_layers, input_size, #Remember that input size is embeddings+hidden_size
                            hidden_size, dropout)

    @property
    def _input_size(self):
        """
        Using input feed by concatenating input with attention vectors.
        """
        return self.embeddings.embedding_size + self.hidden_size


class NMTModel(nn.Module):
    """
    The encoder + decoder Neural Machine Translation Model.
    """
    def __init__(self, encoder, decoder, multigpu=False):
        """
        Args:
            encoder(*Encoder): the various encoder.
            decoder(*Decoder): the various decoder.
            multigpu(bool): run parellel on multi-GPU?
        """
        self.multigpu = multigpu
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, dec_state=None):
        """
        Args:
            src(FloatTensor): a sequence of source tensors with
                    optional feature tensors of size (len x batch).
            tgt(FloatTensor): a sequence of target tensors with
                    optional feature tensors of size (len x batch).
            lengths([int]): an array of the src length.
            dec_state: A decoder state object
        Returns:
            outputs (FloatTensor): (len x batch x hidden_size): decoder outputs
            attns (FloatTensor): Dictionary of (src_len x batch)
            dec_hidden (FloatTensor): tuple (1 x batch x hidden_size)
                                      Init hidden state
        """
        src = src
        tgt = tgt[:-1]  # exclude last target from inputs
        enc_hidden, context = self.encoder(src, lengths)
        enc_state = self.decoder.init_decoder_state(src, context, enc_hidden)
        out, dec_state, attns = self.decoder(tgt, context,
                                             enc_state if dec_state is None
                                             else dec_state)
        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return out, attns, dec_state

class DoubleEncNMTModel(nn.Module):
    """
    The encoder + decoder Neural Machine Translation Model.
    """
    def __init__(self, encoder, decoder, multigpu=False):
        """
        Args:
            encoder(*Encoder): the various encoder.
            decoder(*Decoder): the various decoder.
            multigpu(bool): run parellel on multi-GPU?
        """
        self.multigpu = multigpu
        super(DoubleEncNMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths_src, lengths_inter, dec_state=None):
        """
        Args:
            src(FloatTensor): a sequence of source tensors with
                    optional feature tensors of size (len x batch).
            tgt(FloatTensor): a sequence of target tensors with
                    optional feature tensors of size (len x batch).
            lengths([int]): an array of the src length.
            dec_state: A decoder state object
        Returns:
            outputs (FloatTensor): (len x batch x hidden_size): decoder outputs
            attns (FloatTensor): Dictionary of (src_len x batch)
            dec_hidden (FloatTensor): tuple (1 x batch x hidden_size)
                                      Init hidden state
        """
        src = src
        tgt = tgt[:-1]  # exclude last target from inputs
        enc_src_hidden, src_context, enc_inter_hidden, inter_context= self.encoder(src, lengths_src, lengths_inter)
        #Enc_src_hidden and enc_inter_hidden are the hidden parameters of both encoders
        #src_context and inter context are the encoded sentences (I guess len is the maximum length??)
        src_context=torch.cat((src_context,inter_context),dim=0)
        enc_state = self.decoder.init_decoder_state(src[0], src_context, enc_src_hidden, )
        #print (enc_state)
        #This is to initialize the decoders hidden state!
        #tgt-> y(i-1)  src_context-> s(i)   dec_state-> t(i-1)
        out, dec_state, attns = self.decoder(tgt, src_context,
                                             enc_state if dec_state is None
                                             else dec_state)
        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return out, attns, dec_state


class DecoderState(object):
    """
    DecoderState is a base class for models, used during translation
    for storing translation states.
    """
    def detach(self):
        """
        Detaches all Variables from the graph
        that created it, making it a leaf.
        """
        for h in self._all:
            if h is not None:
                h.detach_()

    def beam_update(self, idx, positions, beam_size):
        """ Update when beam advances. """
        for e in self._all:
            a, br, d = e.size()
            sentStates = e.view(a, beam_size, br // beam_size, d)[:, :, idx]
            sentStates.data.copy_(
                sentStates.data.index_select(1, positions))


class RNNDecoderState(DecoderState):
    def __init__(self, context, hidden_size, rnnstate):
        """
        Args:
            context (FloatTensor): output from the encoder of size
                                   len x batch x rnn_size.
            hidden_size (int): the size of hidden layer of the decoder.
            rnnstate (Variable): final hidden state from the encoder.
                transformed to shape: layers x batch x (directions*dim).
            input_feed (FloatTensor): output from last layer of the decoder.
            coverage (FloatTensor): coverage output from the decoder.
        """
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.coverage = None

        # Init the input feed.
        batch_size = context.size(1)
        h_size = (batch_size, hidden_size)
        #Creates a variable that does not require the update of gradients and gas the size
        self.input_feed = Variable(context.data.new(*h_size).zero_(),
                                   requires_grad=False).unsqueeze(0)

    @property
    def _all(self):
        return self.hidden + (self.input_feed,)

    def update_state(self, rnnstate, input_feed, coverage):
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.input_feed = input_feed
        self.coverage = coverage

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        vars = [Variable(e.data.repeat(1, beam_size, 1), volatile=True)
                for e in self._all]
        self.hidden = tuple(vars[:-1])
        self.input_feed = vars[-1]


class RNNDecoderState_double_enc(DecoderState):
    def __init__(self, context_src, hidden_size_decoder, rnnstate_src, context_inter, rnnstate_inter):
        """
        Args:
            context (FloatTensor): output from the encoder of size
                                   len x batch x rnn_size.
            hidden_size (int): the size of hidden layer of the decoder.
            rnnstate (Variable): final hidden state from the encoder.
                transformed to shape: layers x batch x (directions*dim).
            input_feed (FloatTensor): output from last layer of the decoder.
            coverage (FloatTensor): coverage output from the decoder.
        """
        if not isinstance(rnnstate_src, tuple):
            self.hidden_src = (rnnstate_src,)
        else:
            self.hidden_src = rnnstate_src

        if not isinstance(rnnstate_inter, tuple):
            self.hidden_inter = (rnnstate_inter,)
        else:
            self.hidden_inter = rnnstate_inter
        self.coverage = None

        # Init the input feed.
        # SRC
        batch_size_src = context_src.size(1)
        #INTER
        batch_size_inter = context_inter.size(1)
        aeq(batch_size_src,batch_size_inter)
        h_size= (batch_size_inter, hidden_size_decoder)
        #Creates a variable that does not require the update of gradients and gas the size
        self.input_feed = Variable(context_src.data.new(*h_size).zero_(),
                                   requires_grad=False).unsqueeze(0)

    @property
    def _all(self):
        return self.hidden_src + (self.input_feed,)

    def update_state(self, rnnstate_src, input_feed, coverage):
        if not isinstance(rnnstate_src, tuple):
            self.hidden_src = (rnnstate_src,)
        else:
            self.hidden_src = rnnstate_src
        self.input_feed = input_feed
        self.coverage = coverage

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        vars = [Variable(e.data.repeat(1, beam_size, 1), volatile=True)
                for e in self._all]
        self.hidden_src = tuple(vars[:-1])
        self.input_feed = vars[-1]
