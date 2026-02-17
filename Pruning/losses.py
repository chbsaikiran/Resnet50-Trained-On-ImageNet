# losses.py

import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels,
                      alpha=0.5, T=4):

    ce_loss = F.cross_entropy(student_logits, labels)

    kl_loss = F.kl_div(
        F.log_softmax(student_logits/T, dim=1),
        F.softmax(teacher_logits/T, dim=1),
        reduction='batchmean'
    ) * (T*T)

    return alpha * ce_loss + (1 - alpha) * kl_loss
