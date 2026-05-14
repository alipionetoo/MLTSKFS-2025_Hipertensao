clear all; clc; tic;

fprintf('--- Iniciando Teste ANFIS: DBHA 2025 ---\n');

%% 1. GERAÇÃO DE DADOS SINTÉTICOS (PAS e PAD)
N = 1000; % Número de pacientes sintéticos
% Gera pressões realistas: PAS entre 90 e 200 / PAD entre 60 e 120
PAS = 90 + (200 - 90) * rand(N, 1);
PAD = 60 + (120 - 60) * rand(N, 1);

data_raw = [PAS, PAD]; % Matriz de Entrada (X)

%% 2. CLASSIFICAÇÃO BASEADA NA DIRETRIZ 2025 (Regras Médicas)
target_label = zeros(N, 1);
for i = 1:N
    pas = PAS(i); pad = PAD(i);
    % A regra de ouro da cardiologia: prevalece o pior estágio
    if pas >= 180 || pad >= 110
        target_label(i) = 5; % Estágio 3
    elseif pas >= 160 || pad >= 100
        target_label(i) = 4; % Estágio 2
    elseif pas >= 140 || pad >= 90
        target_label(i) = 3; % Estágio 1
    elseif pas >= 120 || pad >= 80
        target_label(i) = 2; % Pré-Hipertensão
    else
        target_label(i) = 1; % Normal
    end
end

%% 3. PREPARAÇÃO DOS DADOS (One-Hot Encoding e Normalização)
num_classes = 5;
target_onehot = zeros(N, num_classes);
for i = 1:N
    target_onehot(i, target_label(i)) = 1; % Coloca 1 na classe correta
end

% Normaliza as pressões para o intervalo [0,1] matematicamente (Sem toolboxes)
min_val = min(data_raw);
max_val = max(data_raw);
data_norm = (data_raw - min_val) ./ (max_val - min_val);

%% 4. TREINAMENTO DO ANFIS (Fuzzificação e Otimização TSK)
% Hiperparâmetros do modelo
optmParameter = struct('alpha', 0.1, 'beta', 1, 'gamma', 1, ...
    'maxIter', 100, 'minimumLossMargin', 0.01);
TSKoptions = struct('k', 5, 'h', 1); % k=5 regras (5 MFs), como na DBHA 2025

fprintf('Criando funções de pertinência com Fuzzy C-Means...\n');
% Usa o seu arquivo gene_ante_fcm para achar os centros dos sinos
[v, b] = gene_ante_fcm(data_norm, TSKoptions);

% Expansão do espaço de características (Mapeamento Fuzzy - Camada 3 e 4)
% Assumindo que você tem o arquivo calc_x_g no seu diretório
X_g = calc_x_g(data_norm, v, b); 

fprintf('Otimizando os pesos do consequente com ML-TSKFS...\n');
% Chama a nossa nova função Multi-Classe
W_model = ML_TSKFS_MultiClasse(X_g, target_onehot, optmParameter);

%% 5. TESTE E AVALIAÇÃO (Predição)
% Calcula a saída contínua do modelo
Y_pred_continuous = X_g * W_model;

% Argmax: A classe final do paciente é a coluna que obteve a maior pontuação
[~, predicted_labels] = max(Y_pred_continuous, [], 2);

% Calcula a Acurácia Global
acertos = sum(predicted_labels == target_label);
acuracia = (acertos / N) * 100;

fprintf('\n--- RESULTADOS ---\n');
fprintf('Acurácia de Classificação: %.2f%%\n', acuracia);
toc;