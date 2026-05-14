clear all; clc; tic;

fprintf('--- SISTEMA DE SUPORTE À DECISÃO CLÍNICA: DBHA 2025 ---\n');
fprintf('A carregar a Base de Dados Oficial...\n');

%% 1. CARREGAR A BASE DE DADOS FIXA
% Substituímos a geração aleatória (rand) pelo carregamento do ficheiro
load('Base_Pacientes_DBHA.mat'); 

% Extraindo os dados das colunas (5000 pacientes)
PAS = Base_Completa(:, 1);
PAD = Base_Completa(:, 2);
Idade = Base_Completa(:, 3);
Colesterol = Base_Completa(:, 4);
Diabetes = Base_Completa(:, 5);
Tabagismo = Base_Completa(:, 6);
Homem = Base_Completa(:, 7);

% Rótulos (O "Padrão Ouro" do Médico para testarmos o erro da máquina)
Target_Estagio = Base_Completa(:, 8);
Target_Carga = Base_Completa(:, 9);
Target_Risco_Global = Base_Completa(:, 10);

N = length(PAS); % Vai detetar automaticamente os 5000

%% 2. MOTOR 1: ANFIS 1 (Pressão -> Estágio)
fprintf('A treinar o Motor 1 (ANFIS de Pressão)...\n');
Data_A1 = [PAS, PAD];
Data_A1_Norm = (Data_A1 - min(Data_A1)) ./ (max(Data_A1) - min(Data_A1));

T1_OneHot = zeros(N, 5); 
for i=1:N, T1_OneHot(i, Target_Estagio(i))=1; end

opt = struct('alpha', 0.1, 'beta', 1, 'gamma', 1, 'maxIter', 100, 'minimumLossMargin', 0.01);
[v1, b1] = gene_ante_fcm(Data_A1_Norm, struct('k', 15, 'h', 1));
Xg1 = calc_x_g(Data_A1_Norm, v1, b1);
W1 = ML_TSKFS_MultiClasse(Xg1, T1_OneHot, opt);

% O ANFIS 1 faz a sua previsão
[~, Pred_Estagio] = max(Xg1 * W1, [], 2);

%% 3. MOTOR 2: CÁLCULO DE CARGA DE RISCO (PREVENT)
fprintf('A processar o Módulo de Fatores de Risco...\n');
Pred_Carga = zeros(N, 1);
for i = 1:N
    FR = 0;
    if (Homem(i) == 1 && Idade(i) > 55) || (Homem(i) == 0 && Idade(i) > 65), FR = FR + 1; end
    if Colesterol(i) > 190, FR = FR + 1; end
    if Tabagismo(i) == 1, FR = FR + 1; end
    
    if Diabetes(i) == 1 || FR >= 3, Pred_Carga(i) = 2;
    elseif FR >= 1, Pred_Carga(i) = 1;
    else, Pred_Carga(i) = 0; end
end

%% 4. MOTOR 3: ANFIS 2 (Estágio + Carga -> Risco Global)
fprintf('A treinar o Motor 3 (ANFIS de Risco Global)...\n');
% A cascata em ação: O ANFIS 2 usa as previsões dos motores anteriores
Data_A2 = [Pred_Estagio, Pred_Carga]; 
Data_A2_Norm = (Data_A2 - min(Data_A2)) ./ (max(Data_A2) - min(Data_A2));

T2_OneHot = zeros(N, 3); 
for i=1:N, T2_OneHot(i, Target_Risco_Global(i))=1; end

[v2, b2] = gene_ante_fcm(Data_A2_Norm, struct('k', 9, 'h', 1));
Xg2 = calc_x_g(Data_A2_Norm, v2, b2);
W2 = ML_TSKFS_MultiClasse(Xg2, T2_OneHot, opt);

% Previsão Final do Sistema (O Diagnóstico da Máquina)
[~, Pred_Risco_Global] = max(Xg2 * W2, [], 2);

%% 5. MÉTRICAS FINAIS DE AVALIAÇÃO (PARA A DISSERTAÇÃO)
acc_final = (sum(Pred_Risco_Global == Target_Risco_Global) / N) * 100;

fprintf('\n================================================\n');
fprintf('   RESULTADOS FINAIS DA DISSERTAÇÃO (N = %d)    \n', N);
fprintf('================================================\n');
fprintf('Acurácia Global (Ponta a Ponta): %.2f%%\n\n', acc_final);

% Gerando a Matriz de Confusão (Sem usar toolboxes pagas)
Matriz_Confusao = zeros(3, 3); % 3 classes de Risco: Baixo, Moderado, Alto
for i = 1:N
    real = Target_Risco_Global(i);
    previsto = Pred_Risco_Global(i);
    Matriz_Confusao(real, previsto) = Matriz_Confusao(real, previsto) + 1;
end

fprintf('MATRIZ DE CONFUSÃO:\n');
fprintf('                 | Prev: Baixo | Prev: Mod.  | Prev: Alto |\n');
fprintf('-----------------------------------------------------------\n');
fprintf('Real: Risco Baixo|     %4d    |     %4d    |     %4d   |\n', Matriz_Confusao(1,:));
fprintf('Real: Risco Mod. |     %4d    |     %4d    |     %4d   |\n', Matriz_Confusao(2,:));
fprintf('Real: Risco Alto |     %4d    |     %4d    |     %4d   |\n', Matriz_Confusao(3,:));
fprintf('-----------------------------------------------------------\n');
toc;