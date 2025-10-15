from typing import *


class GuidanceIntervalSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance with interval.
    """

    def _inference_model(self, model, x_t, t, cond, cfg_strength, neg_cond, cfg_interval, **kwargs):
        if cfg_interval[0] <= t <= cfg_interval[1]:
            pred = super()._inference_model(model, x_t, t, cond, **kwargs)
            neg_pred = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
            return (1 + cfg_strength) * pred - cfg_strength * neg_pred
        else:
            return super()._inference_model(model, x_t, t, cond, **kwargs)
        
    def _inference_model_style(self, model, x_t1, x_t2, t, cond, cfg_strength, neg_cond, cfg_interval, **kwargs):
        if cfg_interval[0] <= t <= cfg_interval[1]:
            pred = super()._inference_model_style(model, x_t1, x_t2, t, cond, **kwargs)
            neg_pred = super()._inference_model_style(model, x_t1, x_t2, t, neg_cond, **kwargs)
            return (1 + cfg_strength) * pred - cfg_strength * neg_pred
        else:
            return super()._inference_model_style(model, x_t1, x_t2, t, cond, **kwargs)
        
    def _inference_model_style_peserve(self, model, x_t0, x_t1, x_t2, x_t3, t, cond1, cond2, cond3, intensity, cfg_strength, neg_cond, cfg_interval, **kwargs):
        if cfg_interval[0] <= t <= cfg_interval[1]:
            pred_v0, pred_v1, pred_v2, pred_v3 = super()._inference_model_style_peserve(model, x_t0, x_t1, x_t2, x_t3, t, cond1, cond2, cond3, intensity, **kwargs)
            # neg_pred = super()._inference_model_style_peserve(model, x_t1, x_t2, x_t3, t, neg_cond, cond2, **kwargs)
            neg_pred_0 = super()._inference_model(model, x_t0, t, neg_cond, **kwargs)
            neg_pred_1 = super()._inference_model(model, x_t1, t, neg_cond, **kwargs)
            neg_pred_2 = super()._inference_model(model, x_t2, t, neg_cond, **kwargs)
            neg_pred_3 = super()._inference_model(model, x_t3, t, neg_cond, **kwargs)
            return (1 + cfg_strength) * pred_v0 - cfg_strength * neg_pred_0, (1 + cfg_strength) * pred_v1 - cfg_strength * neg_pred_1, (1 + cfg_strength) * pred_v2 - cfg_strength * neg_pred_2, (1 + cfg_strength) * pred_v3 - cfg_strength * neg_pred_3
        else:
            return super()._inference_model_style_peserve(model, x_t0, x_t1, x_t2, x_t3, t, cond1, cond2, cond3, intensity, **kwargs)
        
    def _inference_model_style_mask(self, model, x_t1, x_t2, x_t3, t, cond1, cond2, cond3, mask1, mask2, cfg_strength, neg_cond, cfg_interval, **kwargs):
        if cfg_interval[0] <= t <= cfg_interval[1]:
            pred = super()._inference_model_style_mask(model, x_t1, x_t2, x_t3, t, cond1, cond2, cond3, mask1, mask2, **kwargs)
            # neg_pred = super()._inference_model_style_peserve(model, x_t1, x_t2, x_t3, t, neg_cond, cond2, **kwargs)
            neg_pred = super()._inference_model(model, x_t1, t, neg_cond, **kwargs)
            return (1 + cfg_strength) * pred - cfg_strength * neg_pred
        else:
            return super()._inference_model_style_peserve(model, x_t1, x_t2, x_t3, t, cond1, cond2, cond3, **kwargs)
        
    def _inference_model_nocondfinetune(self, model, x_t1, t, cond1, cfg_strength, neg_cond, cfg_interval, **kwargs):
        if cfg_interval[0] <= t <= cfg_interval[1]:
            pred = super()._inference_model_nocondfinetune(model, x_t1, t, neg_cond, **kwargs)
            # neg_pred = super()._inference_model_nocondfinetune(model, x_t1, t, neg_cond, **kwargs)
            return pred
        else:
            return super()._inference_model_nocondfinetune(model, x_t1, t, cond1, **kwargs)
        
# class GuidanceIntervalSamplerMixin_Style:
#     """
#     A mixin class for samplers that apply classifier-free guidance with interval.
#     """

#     def _inference_model(self, model, x_t, t, cond1, cond2, neg_cond, cfg_strength, cfg_interval, **kwargs):
#         if cfg_interval[0] <= t <= cfg_interval[1]:
#             pred = super()._inference_model(model, x_t, t, cond, **kwargs)
#             neg_pred = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
#             return (1 + cfg_strength) * pred - cfg_strength * neg_pred
#         else:
#             return super()._inference_model(model, x_t, t, cond, **kwargs)
