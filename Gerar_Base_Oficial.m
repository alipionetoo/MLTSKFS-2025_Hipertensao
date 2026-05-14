clear all; clc;

fprintf('--- GERANDO BASE DE DADOS SINTÉTICA OFICIAL (DBHA 2025) ---\n');

N = 5000; % Base robusta com 5.000 pacientes

%% 1. GERANDO AS VARIÁVEIS CLÍNICAS
PAS = 90 + (200 - 90) * rand(N, 1);
PAD = 60 + (120 - 60) * rand(N, 1);
Idade = round(30 + (80 - 30) * rand(N, 1)); 
Colesterol = round(130 + (300 - 130) * rand(N, 1)); 
Diabetes = rand(N, 1) < 0.20; 
Tabagismo = rand(N, 1) < 0.15; 
Homem = rand(N, 1) < 0.50; 

%% 2. ROTULAGEM (O Padrão Ouro do Médico)
Target_Estagio = zeros(N, 1);
Target_Carga = zeros(N, 1);
Target_Risco_Global = zeros(N, 1);

for i = 1:N
    % Estágio da Pressão
    if PAS(i) >= 180 || PAD(i) >= 110, Target_Estagio(i) = 5;
    elseif PAS(i) >= 160 || PAD(i) >= 100, Target_Estagio(i) = 4;
    elseif PAS(i) >= 140 || PAD(i) >= 90, Target_Estagio(i) = 3;
    elseif PAS(i) >= 120 || PAD(i) >= 80, Target_Estagio(i) = 2;
    else, Target_Estagio(i) = 1; end
    
    % Carga de Risco
    FR = 0;
    if (Homem(i) == 1 && Idade(i) > 55) || (Homem(i) == 0 && Idade(i) > 65), FR = FR + 1; end
    if Colesterol(i) > 190, FR = FR + 1; end
    if Tabagismo(i) == 1, FR = FR + 1; end
    if Diabetes(i) == 1 || FR >= 3, Target_Carga(i) = 2;
    elseif FR >= 1, Target_Carga(i) = 1;
    else, Target_Carga(i) = 0; end

    % Risco Global
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

%% 3. MONTANDO A TABELA E SALVANDO
% Junta tudo numa matriz gigante
Base_Completa = [PAS, PAD, Idade, Colesterol, double(Diabetes), double(Tabagismo), double(Homem), Target_Estagio, Target_Carga, Target_Risco_Global];

% Salva no formato MATLAB (.mat) que é super rápido de carregar
save('Base_Pacientes_DBHA.mat', 'Base_Completa');

% (Opcional) Salva em CSV para você poder abrir no Excel e olhar os dados
csvwrite('Base_Pacientes_DBHA.csv', Base_Completa);

fprintf('SUCESSO! Base salva como "Base_Pacientes_DBHA.mat" e "Base_Pacientes_DBHA.csv" na sua pasta.\n');