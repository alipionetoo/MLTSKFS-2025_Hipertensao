clear all; clc; tic;

fprintf('--- SISTEMA DE SUPORTE À DECISÃO CLÍNICA (DBHA 2025) ---\n');

%% 1. GERAR PACIENTES E CARGA DE RISCO (O código que acabou de testar)
N = 1000; % Vamos testar com 1000 pacientes
PAS = 90 + (200 - 90) * rand(N, 1);
PAD = 60 + (120 - 60) * rand(N, 1);
Idade = round(30 + (80 - 30) * rand(N, 1)); 
Colesterol = round(130 + (300 - 130) * rand(N, 1)); 
Diabetes = rand(N, 1) < 0.20; 
Tabagismo = rand(N, 1) < 0.15; 
Homem = rand(N, 1) < 0.50; 

Estagio_Pressao = zeros(N, 1);
Carga_Risco = zeros(N, 1);
Risco_Global = zeros(N, 1);

%% 2. LÓGICA MÉDICA: ROTULAR OS PACIENTES
for i = 1:N
    % A. Determinar Estágio da Pressão (Saída do ANFIS 1)
    if PAS(i) >= 180 || PAD(i) >= 110, Estagio_Pressao(i) = 5;
    elseif PAS(i) >= 160 || PAD(i) >= 100, Estagio_Pressao(i) = 4;
    elseif PAS(i) >= 140 || PAD(i) >= 90, Estagio_Pressao(i) = 3;
    elseif PAS(i) >= 120 || PAD(i) >= 80, Estagio_Pressao(i) = 2;
    else, Estagio_Pressao(i) = 1; end
    
    % B. Determinar Carga de Fatores de Risco
    FR_count = 0;
    if (Homem(i) == 1 && Idade(i) > 55) || (Homem(i) == 0 && Idade(i) > 65), FR_count = FR_count + 1; end
    if Colesterol(i) > 190, FR_count = FR_count + 1; end
    if Tabagismo(i) == 1, FR_count = FR_count + 1; end
    
    if Diabetes(i) == 1 || FR_count >= 3, Carga_Risco(i) = 2;
    elseif FR_count >= 1 && FR_count <= 2, Carga_Risco(i) = 1;
    else, Carga_Risco(i) = 0; end
    
    % C. A MATRIZ DE RISCO CARDIOVASCULAR (O Alvo do ANFIS 2)
    % Cruzamento do Estágio da Pressão vs. Carga de Risco
    if Estagio_Pressao(i) == 1 % Normal
        Risco_Global(i) = 1; % Baixo
    elseif Estagio_Pressao(i) == 2 % Pré-hipertenso
        if Carga_Risco(i) == 2, Risco_Global(i) = 3; % Alto
        else, Risco_Global(i) = 2; end % Moderado
    elseif Estagio_Pressao(i) == 3 % Estágio 1
        if Carga_Risco(i) == 0, Risco_Global(i) = 1; % Baixo
        elseif Carga_Risco(i) == 1, Risco_Global(i) = 2; % Moderado
        else, Risco_Global(i) = 3; end % Alto
    else % Estágio 2 e 3
        if Carga_Risco(i) == 0, Risco_Global(i) = 2; % Moderado
        else, Risco_Global(i) = 3; end % Alto
    end
end

%% 3. PREPARAÇÃO PARA O ANFIS 2
% O ANFIS 2 só recebe 2 colunas de entrada: O Estágio (1 a 5) e a Carga (0 a 2)
Entradas_ANFIS2 = [Estagio_Pressao, Carga_Risco];

% Normalização das entradas para [0,1]
min_val = min(Entradas_ANFIS2);
max_val = max(Entradas_ANFIS2);
Entradas_Norm = (Entradas_ANFIS2 - min_val) ./ (max_val - min_val);

% One-Hot Encoding para as 3 classes de Risco Global (Baixo, Moderado, Alto)
num_classes = 3;
target_onehot = zeros(N, num_classes);
for i = 1:N
    target_onehot(i, Risco_Global(i)) = 1;
end

%% 4. TREINAMENTO DO ANFIS 2
fprintf('A treinar o ANFIS 2 (Matriz de Risco Global)...\n');
optmParameter = struct('alpha', 0.1, 'beta', 1, 'gamma', 1, 'maxIter', 100, 'minimumLossMargin', 0.01);
TSKoptions = struct('k', 9, 'h', 1); % Usaremos 9 regras para cobrir bem os cruzamentos

[v, b] = gene_ante_fcm(Entradas_Norm, TSKoptions);
X_g = calc_x_g(Entradas_Norm, v, b); 
W_model = ML_TSKFS_MultiClasse(X_g, target_onehot, optmParameter);

%% 5. AVALIAÇÃO DA PRECISÃO
Y_pred_continuous = X_g * W_model;
[~, predicted_labels] = max(Y_pred_continuous, [], 2);

acertos = sum(predicted_labels == Risco_Global);
acuracia = (acertos / N) * 100;

fprintf('\n--- RESULTADOS FINAIS DO SISTEMA ---\n');
fprintf('Acurácia de Estratificação de Risco (ANFIS 2): %.2f%%\n', acuracia);
toc;