clear all; clc;

fprintf('--- Iniciando Geração de Pacientes Completos (DBHA 2025) ---\n\n');

N = 10; % Vamos testar com apenas 10 pacientes primeiro para podermos ler a tela

%% 1. GERANDO DADOS CONTÍNUOS (Pressão, Idade, Colesterol)
PAS = 90 + (200 - 90) * rand(N, 1);
PAD = 60 + (120 - 60) * rand(N, 1);

% Idade: Pacientes de 30 a 80 anos
Idade = round(30 + (80 - 30) * rand(N, 1)); 

% Colesterol Total: De 130 a 300 mg/dL
Colesterol = round(130 + (300 - 130) * rand(N, 1)); 

%% 2. GERANDO DADOS BINÁRIOS (0 = Não, 1 = Sim)
% Sorteia se o paciente tem Diabetes (20% de chance)
Diabetes = rand(N, 1) < 0.20; 

% Sorteia se o paciente Fuma (15% de chance)
Tabagismo = rand(N, 1) < 0.15; 

% Sorteia Sexo (0 = Mulher, 1 = Homem)
Homem = rand(N, 1) < 0.50; 

%% 3. MÓDULO DE FATORES DE RISCO (A Lógica da DBHA 2025)
% Aqui vamos contar quantos Fatores de Risco (FR) o paciente tem
Carga_Risco = zeros(N, 1); 

for i = 1:N
    FR_count = 0;
    
    % Regras de Fatores de Risco da DBHA:
    if (Homem(i) == 1 && Idade(i) > 55) || (Homem(i) == 0 && Idade(i) > 65)
        FR_count = FR_count + 1;
    end
    if Colesterol(i) > 190
        FR_count = FR_count + 1;
    end
    if Tabagismo(i) == 1
        FR_count = FR_count + 1;
    end
    
    % Classificação da Carga de Risco (0, 1 ou 2)
    % A DBHA 2025 diz que se tiver Diabetes, o risco já pula pro teto.
    if Diabetes(i) == 1 || FR_count >= 3
        Carga_Risco(i) = 2; % Risco Alto (Muitos fatores ou Doença Crônica)
    elseif FR_count >= 1 && FR_count <= 2
        Carga_Risco(i) = 1; % Risco Moderado (1 ou 2 fatores)
    else
        Carga_Risco(i) = 0; % Sem Fatores de Risco
    end
end

%% 4. EXIBINDO OS RESULTADOS PARA CONFERÊNCIA
fprintf(' ID | PAS / PAD  | Idade | Colest. | Fuma | Diab. || CARGA DE RISCO\n');
fprintf('--------------------------------------------------------------------\n');
for i = 1:N
    fprintf(' %02d | %3.0f / %3.0f |  %2.0f   |   %3.0f   |   %d  |   %d   ||      %d\n', ...
        i, PAS(i), PAD(i), Idade(i), Colesterol(i), Tabagismo(i), Diabetes(i), Carga_Risco(i));
end

% A Matriz Final de Dados que vai alimentar os ANFIS:
% Colunas: [PAS, PAD, Idade, Colesterol, Tabagismo, Diabetes, Carga_Risco]
Base_Pacientes = [PAS, PAD, Idade, Colesterol, Tabagismo, Diabetes, Carga_Risco];