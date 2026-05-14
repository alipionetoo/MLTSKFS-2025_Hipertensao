clear all; clc; tic;

fprintf('--- INICIANDO SISTEMA HÍBRIDO EM CASCATA (DBHA 2025) ---\n');

%% 1. GERAÇÃO E ROTULAGEM (GROUND TRUTH)
N = 1500; % Aumentamos para 1500 para um treino mais robusto
PAS = 90 + (200 - 90) * rand(N, 1);
PAD = 60 + (120 - 60) * rand(N, 1);
Idade = round(30 + (80 - 30) * rand(N, 1)); 
Colesterol = round(130 + (300 - 130) * rand(N, 1)); 
Diabetes = rand(N, 1) < 0.20; 
Tabagismo = rand(N, 1) < 0.15; 
Homem = rand(N, 1) < 0.50; 

% Matrizes para guardar o "Padrão Ouro" (Médico)
Target_Estagio = zeros(N, 1);
Target_Carga = zeros(N, 1);
Target_Risco_Global = zeros(N, 1);

for i = 1:N
    % Rótulo Estágio (Alvo ANFIS 1)
    if PAS(i) >= 180 || PAD(i) >= 110, Target_Estagio(i) = 5;
    elseif PAS(i) >= 160 || PAD(i) >= 100, Target_Estagio(i) = 4;
    elseif PAS(i) >= 140 || PAD(i) >= 90, Target_Estagio(i) = 3;
    elseif PAS(i) >= 120 || PAD(i) >= 80, Target_Estagio(i) = 2;
    else, Target_Estagio(i) = 1; end
    
    % Rótulo Carga de Risco (Módulo PREVENT)
    FR = 0;
    if (Homem(i) == 1 && Idade(i) > 55) || (Homem(i) == 0 && Idade(i) > 65), FR = FR + 1; end
    if Colesterol(i) > 190, FR = FR + 1; end
    if Tabagismo(i) == 1, FR = FR + 1; end
    if Diabetes(i) == 1 || FR >= 3, Target_Carga(i) = 2;
    elseif FR >= 1, Target_Carga(i) = 1;
    else, Target_Carga(i) = 0; end

    % Rótulo Risco Global (Alvo ANFIS 2)
    if Target_Estagio(i) == 1, Target_Risco_Global(i) = 1;
    elseif Target_Estagio(i) == 2
        if Target_Carga(i) == 2, Target_Risco_Global(i) = 3; else, Target_Risco_Global(i) = 2; end
    elseif Target_Estagio(i) == 3
        if Target_Carga(i) == 0, Target_Risco_Global(i) = 1; 
        elseif Target_Carga(i) == 1, Target_Risco_Global(i) = 2; else, Target_Risco_Global(i) = 3; end
    else
        if Target_Carga(i) == 0, Target_Risco_Global(i) = 2; else, Target_Risco_Global(i) = 3; end
    end
end

%% 2. TREINAMENTO DO ANFIS 1 (Pressão -> Estágio)
fprintf('Treinando ANFIS 1...\n');
Data_A1 = [PAS, PAD];
Data_A1_Norm = (Data_A1 - min(Data_A1)) ./ (max(Data_A1) - min(Data_A1));
T1_OneHot = zeros(N, 5); for i=1:N, T1_OneHot(i, Target_Estagio(i))=1; end

opt1 = struct('alpha', 0.1, 'beta', 1, 'gamma', 1, 'maxIter', 100, 'minimumLossMargin', 0.01);
[v1, b1] = gene_ante_fcm(Data_A1_Norm, struct('k', 15, 'h', 1));
Xg1 = calc_x_g(Data_A1_Norm, v1, b1);
W1 = ML_TSKFS_MultiClasse(Xg1, T1_OneHot, opt1);

%% 3. TREINAMENTO DO ANFIS 2 (Estágio + Carga -> Risco)
fprintf('Treinando ANFIS 2...\n');
% Simulando a entrada do ANFIS 2 usando o que o ANFIS 1 previu
[~, Pred_Estagio] = max(Xg1 * W1, [], 2);
Data_A2 = [Pred_Estagio, Target_Carga];
Data_A2_Norm = (Data_A2 - min(Data_A2)) ./ (max(Data_A2) - min(Data_A2));
T2_OneHot = zeros(N, 3); for i=1:N, T2_OneHot(i, Target_Risco_Global(i))=1; end

[v2, b2] = gene_ante_fcm(Data_A2_Norm, struct('k', 9, 'h', 1));
Xg2 = calc_x_g(Data_A2_Norm, v2, b2);
W2 = ML_TSKFS_MultiClasse(Xg2, T2_OneHot, opt1);

%% 4. TESTE FINAL DO SISTEMA COMPLETO (End-to-End Accuracy)
[~, Final_Pred] = max(Xg2 * W2, [], 2);
acc_final = (sum(Final_Pred == Target_Risco_Global) / N) * 100;

fprintf('\n--- RESULTADOS DA INTEGRAÇÃO ---\n');
fprintf('Acurácia Final do Sistema (Ponta a Ponta): %.2f%%\n', acc_final);
toc;