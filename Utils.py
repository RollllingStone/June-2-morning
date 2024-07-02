import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.Models import get_non_pad_mask


def softplus(x, beta):
    # hard thresholding at 20
    temp = beta * x
    temp[temp > 20] = 20
    return 1.0 / beta * torch.log(1 + torch.exp(temp))


def compute_event(event, non_pad_mask):
    """ Log-likelihood of events. """

    # add 1e-9 in case some events have 0 likelihood
    event += math.pow(10, -9)
    event.masked_fill_(~non_pad_mask.bool(), 1.0)

    result = torch.log(event)
    return result


def compute_integral_biased(all_lambda, time, non_pad_mask):
    """ Log-likelihood of non-events, using linear interpolation. """

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
    diff_lambda = (all_lambda[:, 1:] + all_lambda[:, :-1]) * non_pad_mask[:, 1:]

    biased_integral = diff_lambda * diff_time
    result = 0.5 * biased_integral
    return result


def compute_integral_unbiased(model, data, time, non_pad_mask, type_mask):
    """ Log-likelihood of non-events, using Monte Carlo integration. """

    num_samples = 100

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
    temp_time = diff_time.unsqueeze(2) * \
                torch.rand([*diff_time.size(), num_samples], device=data.device)
    temp_time /= (time[:, :-1] + 1).unsqueeze(2)

    temp_hid = model.linear(data)[:, 1:, :]
    temp_hid = torch.sum(temp_hid * type_mask[:, 1:, :], dim=2, keepdim=True)

    all_lambda = softplus(temp_hid + model.alpha * temp_time, model.beta)
    all_lambda = torch.sum(all_lambda, dim=2) / num_samples

    unbiased_integral = all_lambda * diff_time
    return unbiased_integral


def compute_integral_unbiased(model, data, time, non_pad_mask, type_mask):
    """ Log-likelihood of non-events, using Monte Carlo integration. """

    # data是enc_output
    '''现在是enc_ouptut是一个store的  
    我们要的enc_output也是一个store的 就只有一个center store的hidden representation

    那这个store_index是个vector  对应batch中每个store的index

    # enc_output: batch * seq_len * model_dim
    '''
    num_samples = 100

    #这一段是t-tj  算距离上个event过去了多久
    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
    temp_time = diff_time.unsqueeze(2) * \
                torch.rand([*diff_time.size(), num_samples], device=data.device)
    temp_time /= (time[:, :-1] + 1).unsqueeze(2)

    #这一段是算我们的h_i
    #第一个没法predict


    temp_hid = torch.matmul(data, model.w)
    # all_hid shape :[batch_size, seq_len, num_types]
    #advanced indexing, self.b_lamda[store_idx] should be shape (batch_size, 1)
    #unsqueeze后 shape:  (batch_size, 1, 1)
    temp_hid += model.bias.unsqueeze(1).unsqueeze(2)
    temp_hid = temp_hid[:, 1:, :] 

    # temp_hid = model.linear(data)[:, 1:, :]
    temp_hid = torch.sum(temp_hid * type_mask[:, 1:, :], dim=2, keepdim=True)

    #beta是softplus的beta 和b_i没有关系 要加b_i在这加  v也在这加
    all_lambda = softplus(temp_hid + model.alpha * temp_time, model.beta)
    all_lambda = torch.sum(all_lambda, dim=2) / num_samples

    unbiased_integral = all_lambda * diff_time
    return unbiased_integral

# def compute_integral_unbiased(model, data, time, non_pad_mask, type_mask, store_idx):
#     """ Log-likelihood of non-events, using Monte Carlo integration. """

#     # data是enc_output
#     '''现在是enc_ouptut是一个store的  
#     我们要的enc_output也是一个store的 就只有一个center store的hidden representation

#     那这个store_index是个vector  对应batch中每个store的index

#     # enc_output: batch * seq_len * model_dim
#     '''
#     num_samples = 100

#     #这一段是t-tj  算距离上个event过去了多久
#     diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
#     temp_time = diff_time.unsqueeze(2) * \
#                 torch.rand([*diff_time.size(), num_samples], device=data.device)
#     temp_time /= (time[:, :-1] + 1).unsqueeze(2)

#     #这一段是算我们的h_i
#     #第一个没法predict


#     temp_hid = torch.matmul(enc_output, self.v_lamda)
#     # all_hid shape :[batch_size, seq_len, num_types]
#     #advanced indexing, self.b_lamda[store_idx] should be shape (batch_size, 1)
#     #unsqueeze后 shape:  (batch_size, 1, 1)
#     temp_hid += self.b_lamda[store_idx].unsqueeze(1).unsqueeze(2)
#     temp_hid = temp_hid[:, 1:, :] 

#     # temp_hid = model.linear(data)[:, 1:, :]
#     temp_hid = torch.sum(temp_hid * type_mask[:, 1:, :], dim=2, keepdim=True)

#     #beta是softplus的beta 和b_i没有关系 要加b_i在这加  v也在这加
#     all_lambda = softplus(temp_hid + model.alpha * temp_time, model.beta)
#     all_lambda = torch.sum(all_lambda, dim=2) / num_samples

#     unbiased_integral = all_lambda * diff_time
#     return unbiased_integral

def gamma_pdf(m, alpha, beta):
    """Compute the PDF of the Gamma distribution."""
    ## computed element-wise

    gamma_alpha = torch.exp(torch.lgamma(alpha))
    return (beta ** alpha) / gamma_alpha * (m ** (alpha - 1)) * torch.exp(-beta * m)

def log_likelihood(model, enc_output, time, types, true_demand):
    """ Log-likelihood of sequence. """

    # enc_output: batch * seq_len * model_dim
    #就是一个batch算一个log likelihood

    #在这里我们就是针对一个lamda_i在算的  一次肯定只有一个store啊  只不过有batch size个store

    #store_idx shape: (batch_size, 1)

    #pad 是batch一起pad

    #true_demand shape: (batch_size, seq_length)  --- seq_length: max_length among batch

    non_pad_mask = get_non_pad_mask(types).squeeze(2)

    type_mask = torch.zeros([*types.size(), model.num_types], device=enc_output.device)
    for i in range(model.num_types):
        type_mask[:, :, i] = (types == i + 1).bool().to(enc_output.device)


    #1.  event log-likelihood

    # all_hid = model.linear_lamda(enc_output)
    all_hid = torch.matmul(enc_output, model.w)
    # all_hid shape :[batch_size, seq_len, num_types]
    all_hid += model.bias.unsqueeze(1).unsqueeze(2)
    all_lambda = softplus(all_hid, model.beta)
    #这个是算total lamda rate, 我们一个type  不影响,  
    type_lambda = torch.sum(all_lambda * type_mask, dim=2)
    #type_lamda shape: [batch_size, seq_len]

    #这行就是去算了个log
    event_ll1 = compute_event(type_lambda, non_pad_mask)
    event_ll1 = torch.sum(event_ll1, dim=-1)

   
    #2. event log-likelihood Gamma

    hid_v_alpha = torch.matmul(enc_output, model.v_alpha)
    hid_v_alpha += model.b_alpha.unsqueeze(1).unsqueeze(2)
    all_alpha = softplus(hid_v_alpha, model.beta)
    #这个是算total lamda rate, 我们一个type  不影响
    type_alpha = torch.sum(all_alpha * type_mask, dim=2)
    #type_alpha shape: [batch_size, seq_len]

    hid_v_beta = torch.matmul(enc_output, model.v_beta)
    hid_v_beta += model.b_beta.unsqueeze(1).unsqueeze(2)
    all_beta = softplus(hid_v_beta, model.beta)
    #这个是算total lamda rate, 我们一个type  不影响
    type_beta = torch.sum(all_beta * type_mask, dim=2)
    #type_alpha_i, type_beta_i 就是我们的alpha和beta, shape: [batch_size, seq_len]
    pdf_gamma = gamma_pdf(true_demand, type_alpha, type_beta)

    event_ll2 = compute_event(pdf_gamma, non_pad_mask)
    event_ll2 = torch.sum(event_ll2, dim=-1)

    event_ll = event_ll1 + event_ll2

    #3. non-event log-likelihood, either numerical integration or MC integration
    # non_event_ll = compute_integral_biased(type_lambda, time, non_pad_mask)
    non_event_ll = compute_integral_unbiased(model, enc_output, time, non_pad_mask, type_mask)
    non_event_ll = torch.sum(non_event_ll, dim=-1)

    return event_ll, non_event_ll

# def log_likelihood(model, enc_output, time, types, store_idx, true_demand):
#     """ Log-likelihood of sequence. """

#     # enc_output: batch * seq_len * model_dim
#     #就是一个batch算一个log likelihood

#     #在这里我们就是针对一个lamda_i在算的  一次肯定只有一个store啊  只不过有batch size个store

#     #store_idx shape: (batch_size, 1)

#     #pad 是batch一起pad

#     #true_demand shape: (batch_size, seq_length)  --- seq_length: max_length among batch

#     non_pad_mask = get_non_pad_mask(types).squeeze(2)

#     type_mask = torch.zeros([*types.size(), model.num_types], device=enc_output.device)
#     for i in range(model.num_types):
#         type_mask[:, :, i] = (types == i + 1).bool().to(enc_output.device)


#     #1.  event log-likelihood

#     # all_hid = model.linear_lamda(enc_output)
#     all_hid = torch.matmul(enc_output, self.v_lamda)
#     # all_hid shape :[batch_size, seq_len, num_types]
#     #advanced indexing, self.b_lamda[store_idx] should be shape (batch_size, 1)
#     #unsqueeze后 shape:  (batch_size, 1, 1)
#     all_hid += self.b_lamda[store_idx].unsqueeze(1).unsqueeze(2)
#     all_lambda = softplus(all_hid, model.beta)
#     #这个是算total lamda rate, 我们一个type  不影响,  
#     type_lambda = torch.sum(all_lambda * type_mask, dim=2)
#     #type_lamda shape: [batch_size, seq_len]

#     #这行就是去算了个log
#     event_ll1 = compute_event(type_lambda, non_pad_mask)
#     event_ll1 = torch.sum(event_ll1, dim=-1)

   
#     #2. event log-likelihood Gamma

#     hid_v_alpha = torch.matmul(enc_output, self.v_alpha)
#     hid_v_alpha += self.b_alpha[store_idx].unsqueeze(1).unsqueeze(2)
#     all_alpha = softplus(hid_v_alpha, model.beta)
#     #这个是算total lamda rate, 我们一个type  不影响
#     type_alpha = torch.sum(all_alpha * type_mask, dim=2)
#     #type_alpha shape: [batch_size, seq_len]

#     hid_v_beta = torch.matmul(enc_output, self.v_beta)
#     hid_v_beta += self.b_beta[store_idx].unsqueeze(1).unsqueeze(2)
#     all_beta = softplus(hid_v_beta, model.beta)
#     #这个是算total lamda rate, 我们一个type  不影响
#     type_beta = torch.sum(all_beta * type_mask, dim=2)
#     #type_alpha_i, type_beta_i 就是我们的alpha和beta, shape: [batch_size, seq_len]
#     pdf_gamma = gamma_pdf(true_demand, type_alpha, type_beta)

#     event_ll2 = compute_event(pdf_gamma, non_pad_mask)
#     event_ll2 = torch.sum(event_ll2, dim=-1)

#     event_ll = event_ll1 + event_ll2

#     #3. non-event log-likelihood, either numerical integration or MC integration
#     # non_event_ll = compute_integral_biased(type_lambda, time, non_pad_mask)
#     non_event_ll = compute_integral_unbiased(model, enc_output, time, non_pad_mask, type_mask, store_idx)
#     non_event_ll = torch.sum(non_event_ll, dim=-1)

#     return event_ll, non_event_ll

def type_loss(prediction, types, loss_func):
    """ Event prediction loss, cross entropy or label smoothing. """

    # convert [1,2,3] based types to [0,1,2]; also convert padding events to -1
    truth = types[:, 1:] - 1
    prediction = prediction[:, :-1, :]

    pred_type = torch.max(prediction, dim=-1)[1]
    correct_num = torch.sum(pred_type == truth)

    # compute cross entropy loss
    if isinstance(loss_func, LabelSmoothingLoss):
        loss = loss_func(prediction, truth)
    else:
        loss = loss_func(prediction.transpose(1, 2), truth)

    loss = torch.sum(loss)
    return loss, correct_num


def time_loss(prediction, event_time):
    """ Time prediction loss. """

    prediction.squeeze_(-1)

    true = event_time[:, 1:] - event_time[:, :-1]
    #在最后一个位置上predict的没法验证
    prediction = prediction[:, :-1]

    # event time gap prediction
    diff = prediction - true
    se = torch.sum(diff * diff)
    return se


def demand_loss(prediction, true_demand):
    """ Time prediction loss. """

    prediction.squeeze_(-1)

    diff = prediction - true_demand
    se = torch.sum(diff * diff)
    return se

class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()

        self.eps = label_smoothing
        self.num_classes = tgt_vocab_size
        self.ignore_index = ignore_index

    def forward(self, output, target):
        """
        output (FloatTensor): (batch_size) x n_classes
        target (LongTensor): batch_size
        """

        non_pad_mask = target.ne(self.ignore_index).float()

        target[target.eq(self.ignore_index)] = 0
        one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / self.num_classes

        log_prb = F.log_softmax(output, dim=-1)
        loss = -(one_hot * log_prb).sum(dim=-1)
        loss = loss * non_pad_mask
        return loss
