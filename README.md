# VLBA-Bench
Multimodal Large Language Models (MLLMs) have made continuous progress in video language understand-
ing tasks. However, since such models usually rely on large-scale open data for training, even at data poisoning
ratios as low as 0.1%–10%, they may still be implanted with backdoors that trigger targeted anomalous out-
puts during the inference phase. Existing backdoor studies mainly focus on image or pure text scenarios, while
systematic evaluations for video language tasks remain insufficient. To this end, we propose VLBA-Bench, the
first systematic evaluation benchmark for backdoor attacks on video language models. The benchmark uniformly
evaluates four categories of backdoor attack paradigms on three video language datasets and one long-video lan-
guage model. In the system experiments, we evaluate more than 50 configurations with different attack paradigms,
target words, and poisoning ratios. The experimental results show that video language models exhibit signif-
icantly different backdoor behaviors across modalities: time-distributed trigger-based dirty-label video attacks
achieve high stealth while maintaining relatively strong clean performance; text-based trigger attacks can reach
high attack success rates under extremely low poisoning ratios, but substantially degrade performance on clean
samples. In contrast, clean-label attacks and static 2D video triggers are considerably less effective in poisoning.
This work systematically reveals, for the first time, the modality differences and time-dependent characteristics
of backdoor attacks in video-language models, and demonstrates that security evaluation of video language mod-
els must explicitly account for multimodal and temporal interaction mechanisms. 
Our code will be comming soon.
