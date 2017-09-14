%./../../dataset/WSJ0/multi_spk_selected_8kHz/test/spk10_205/205a0103.wav,./../../dataset/WSJ0/multi_spk_selected_8kHz/test/spk06_407/407a010o.wav,./../../dataset/WSJ0/multi_spk_selected_8kHz/test/spk09_204/204a010l.wav,./../../dataset/WSJ0/multi_spk_selected_8kHz/test/spk05_406/406o030v.wav,./../../dataset/WSJ0/multi_spk_selected_8kHz/test/spk03_404/404c020g.wav,./../../dataset/WSJ0/multi_spk_selected_8kHz/test/spk07_408/408a0104.wav spk01
clear
clc
max_len = 50000;
true_signal = audioread('./../../dataset/WSJ0/multi_spk_selected_8kHz/test/spk01_401/401a010k.wav');
true_noise_0 = audioread('./../../dataset/WSJ0/multi_spk_selected_8kHz/test/spk02_403/403a0101.wav');
true_noise_1 = audioread('./../../dataset/WSJ0/multi_spk_selected_8kHz/test/spk04_405/405c021f.wav');
true_noise_2 = audioread('./../../dataset/WSJ0/multi_spk_selected_8kHz/test/spk08_409/409c020l.wav');

signal = true_signal;
if length(signal) > max_len
    signal = signal(1:max_len);
elseif length(signal) < max_len
    signal(length(signal)+1:max_len) = 0;
end
signal = signal ./sqrt(sum(signal .^2));
true_signal = signal;

signal = true_noise_0;
if length(signal) > max_len
    signal = signal(1:max_len);
elseif length(signal) < max_len
    signal(length(signal)+1:max_len) = 0;
end
signal = signal ./sqrt(sum(signal .^2));
true_noise_0 = signal;

signal = true_noise_1;
if length(signal) > max_len
    signal = signal(1:max_len);
elseif length(signal) < max_len
    signal(length(signal)+1:max_len) = 0;
end
signal = signal ./sqrt(sum(signal .^2));
true_noise_1 = signal;

signal = true_noise_2;
if length(signal) > max_len
    signal = signal(1:max_len);
elseif length(signal) < max_len
    signal(length(signal)+1:max_len) = 0;
end
signal = signal ./sqrt(sum(signal .^2));
true_noise_2 = signal;

true_noise = true_noise_0 + true_noise_1 + true_noise_2;
% true_noise = true_noise ./sqrt(sum(true_noise .^2));

mix = true_signal + true_noise;

BSS_EVAL(true_signal', true_noise', mix', mix')

BSS_EVAL(true_signal', true_noise', true_signal', mix')
