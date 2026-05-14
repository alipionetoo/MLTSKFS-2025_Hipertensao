function [v, b] = gene_ante_fcm(data, options)
    k = options.k;
    h = options.h;
    [n_examples, d] = size(data);

    %% SUBSTITUTO DO 'fcm' (Fuzzy C-Means Manual sem Toolboxes)
    expo = 2;          % Expoente fuzzy (m)
    max_iter = 100;    % Máximo de iterações permitidas
    min_impro = 1e-6;  % Critério de parada (tolerância)

    % 1. Inicializa a matriz de pertinência U aleatoriamente
    U = rand(k, n_examples);
    U = U ./ sum(U, 1); % Normaliza para que a soma de cada coluna seja 1

    for iter = 1:max_iter
        mf = U.^expo; % Matriz fuzzyficada
        
        % 2. Calcula os centros dos clusters (v)
        v = (mf * data) ./ sum(mf, 2);

        % 3. Calcula a distância (ao quadrado) de cada ponto para cada centro
        dist_sq = zeros(k, n_examples);
        for j = 1:k
            dist_sq(j, :) = sum((data - repmat(v(j, :), n_examples, 1)).^2, 2)';
        end

        % 4. Atualiza a matriz U baseado nas distâncias
        tmp = (dist_sq + eps).^(-1 / (expo - 1)); % eps evita divisão por zero
        U_new = tmp ./ sum(tmp, 1);

        % 5. Verifica se o algoritmo convergiu (parada antecipada)
        if max(abs(U_new(:) - U(:))) < min_impro
            U = U_new;
            break;
        end
        U = U_new;
    end
    %% FIM DO FUZZY C-MEANS MANUAL

    % Calcula as larguras dos sinos (b) originais do algoritmo
    b = zeros(k, d);
    for i = 1:k
        v1 = repmat(v(i,:), n_examples, 1);
        u = U(i,:);
        uu = repmat(u', 1, d);
        b(i,:) = sum((data - v1).^2 .* uu, 1) ./ sum(uu, 1);
    end
    b = b * h;
end