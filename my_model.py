��
l��F� j�P.�M�.�}q (X   protocol_versionqM�X   little_endianq�X
   type_sizesq}q(X   shortqKX   intqKX   longqKuu.�(X   moduleq c__main__
ANN
qX   nn.pyqX  class ANN(torch.nn.Module):
	def __init__(self):
		super(ANN, self).__init__()
		
		self.fc1 = torch.nn.Linear(2, 3)
		self.fc2 = torch.nn.Linear(3, 1)
		
	def forward(self, x):
		x = torch.sigmoid(self.fc1(x))
		x = torch.sigmoid(self.fc2(x))
		
		return x
qtqQ)�q}q(X   _backendqctorch.nn.backends.thnn
_get_thnn_function_backend
q)Rq	X   _parametersq
ccollections
OrderedDict
q)RqX   _buffersqh)RqX   _backward_hooksqh)RqX   _forward_hooksqh)RqX   _forward_pre_hooksqh)RqX   _state_dict_hooksqh)RqX   _load_state_dict_pre_hooksqh)RqX   _modulesqh)Rq(X   fc1q(h ctorch.nn.modules.linear
Linear
qXa   C:\Users\User\AppData\Local\Programs\Python\Python37\lib\site-packages\torch\nn\modules\linear.pyqX�	  class Linear(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
qtqQ)�q }q!(hh	h
h)Rq"(X   weightq#ctorch._utils
_rebuild_parameter
q$ctorch._utils
_rebuild_tensor_v2
q%((X   storageq&ctorch
FloatStorage
q'X   2245079740544q(X   cpuq)KNtq*QK KK�q+KK�q,�h)Rq-tq.Rq/�h)Rq0�q1Rq2X   biasq3h$h%((h&h'X   2245079741888q4h)KNtq5QK K�q6K�q7�h)Rq8tq9Rq:�h)Rq;�q<Rq=uhh)Rq>hh)Rq?hh)Rq@hh)RqAhh)RqBhh)RqChh)RqDX   trainingqE�X   in_featuresqFKX   out_featuresqGKubX   fc2qHh)�qI}qJ(hh	h
h)RqK(h#h$h%((h&h'X   2245079740736qLh)KNtqMQK KK�qNKK�qO�h)RqPtqQRqR�h)RqS�qTRqUh3h$h%((h&h'X   2245079749280qVh)KNtqWQK K�qXK�qY�h)RqZtq[Rq\�h)Rq]�q^Rq_uhh)Rq`hh)Rqahh)Rqbhh)Rqchh)Rqdhh)Rqehh)RqfhE�hFKhGKubuhE�ub.�]q (X   2245079740544qX   2245079740736qX   2245079741888qX   2245079749280qe.       �����P���>�@�wx@S�%=ğB�       �������;�?       �!@S���o��>       ��K@