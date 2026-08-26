clear all; clc; tic;

fprintf('--- TREINAMENTO: ANFIS-PREVENT (RISCO CARDIOMETABÓLICO) ---\n');

%% 1. GERAÇÃO DE DADOS (Simulação Avançada)
N = 2000; % Número de pacientes
Idade = 30 + (80 - 30) * rand(N, 1);
Sexo = rand(N, 1) < 0.5; % 0 Mulher, 1 Homem
Colesterol = 130 + (320 - 130) * rand(N, 1);
HDL = 20 + (80 - 20) * rand(N, 1);
TFGe = 15 + (120 - 15) * rand(N, 1); % Função Renal
HbA1c = 4.0 + (12.0 - 4.0) * rand(N, 1); % Diabetes
Tabagismo = rand(N, 1) < 0.15;

%% 2. O ALVO (O GABARITO DO "PREVENT" ORIGINAL)
% Aqui simulamos uma fórmula contínua de risco (0 a 100%) 
% O seu engenheiro de dados fará a fórmula original da AHA aqui.
Risco_Prevent_Real = zeros(N, 1);
for i = 1:N
    % Uma equação complexa simulada de risco:
    risco_base = (Idade(i)/80)*0.3 + (Colesterol(i)/320)*0.2 - (HDL(i)/80)*0.15 + ((120-TFGe(i))/120)*0.25 + (HbA1c(i)/12)*0.2;
    if Tabagismo(i) == 1, risco_base = risco_base + 0.15; end
    if Sexo(i) == 1, risco_base = risco_base * 1.1; end % Homens têm leve agravante
    
    % Limita o risco entre 0 (0%) e 1 (100%)
    Risco_Prevent_Real(i) = max(0, min(1, risco_base));
end

%% 3. PREPARAÇÃO PARA O ANFIS
% Entradas do Módulo Metabólico (Repare: Não tem Pressão Arterial aqui!)
Entradas = [Idade, Sexo, Colesterol, HDL, TFGe, HbA1c, double(Tabagismo)];

% Normalização dos Dados (Obrigatório para o Gradiente)
Min_E = min(Entradas);
Max_E = max(Entradas);
Entradas_Norm = (Entradas - Min_E) ./ (Max_E - Min_E);

Target_Continuo = Risco_Prevent_Real; % Nosso gabarito

%% 4. TREINAMENTO DO ANFIS-PREVENT (Modo Regressão)
fprintf('A otimizar as regras Fuzzy para o eixo Metabólico/Renal...\n');

% Como são 7 variáveis, 15 regras (clusters) são suficientes para mapear a não-linearidade
TSKoptions = struct("k", 15, 'h', 1); 
[v, b] = gene_ante_fcm(Entradas_Norm, TSKoptions);
X_g = calc_x_g(Entradas_Norm, v, b);

% Parâmetros do otimizador
opt = struct('alpha', 0.1, 'beta', 1, 'gamma', 1, 'maxIter', 100, 'minimumLossMargin', 0.01);

% Usamos o ML_TSKFS clássico para prever o valor contínuo
W_Prevent = ML_TSKFS(X_g, Target_Continuo, opt);

%% 5. TESTE E COMPARAÇÃO: ANFIS vs PREVENT ORIGINAL
Previsao_ANFIS = X_g * W_Prevent;

% Impede que a previsão matemática da IA saia do limite realista (0 a 100%)
Previsao_ANFIS = max(0, min(1, Previsao_ANFIS));

% Cálculo do Erro Médio (Root Mean Square Error) e MAE (Mean Absolute Error)
RMSE = sqrt(mean((Target_Continuo - Previsao_ANFIS).^2));
MAE = mean(abs(Target_Continuo - Previsao_ANFIS));

fprintf('\n================================================\n');
fprintf('   RESULTADOS DO MOTOR PREVENT (Regressão)\n');
fprintf('================================================\n');
fprintf('O algoritmo aprendeu a fórmula complexa com:\n');
fprintf('Erro Absoluto Médio (MAE): %.4f (%.2f%% de desvio)\n', MAE, MAE*100);
fprintf('Erro Quadrático Médio (RMSE): %.4f\n\n', RMSE);

% Demonstração prática no terminal de 5 pacientes aleatórios
fprintf('--- Comparação em 5 Pacientes Aleatórios ---\n');
fprintf('Paciente | PREVENT Oficial |  ANFIS-PREVENT  | Diferença\n');
for i = 1:5
    idx = randi(N);
    fprintf('  %04d   |      %5.2f%%     |      %5.2f%%     |  %5.2f%%\n', ...
        idx, Target_Continuo(idx)*100, Previsao_ANFIS(idx)*100, abs(Target_Continuo(idx)-Previsao_ANFIS(idx))*100);
end
toc;