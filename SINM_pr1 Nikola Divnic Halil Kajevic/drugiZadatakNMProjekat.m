clc, clear, close all

%% Ucitavanje podataka
load dataset2.mat

ob = pod(:, 1:2)';
klasa = pod(:, 3)';
N = length(klasa);

K1 = ob(:, klasa == 1);
K2 = ob(:, klasa == 2);
K3 = ob(:, klasa == 3);

figure, hold all
plot(K1(1, :), K1(2, :), 'o')
plot(K2(1, :), K2(2, :), '*')
plot(K3(1, :), K3(2, :), 'd')

%% One-hot encoding
izlaz = zeros(3, N);

izlaz(1, klasa == 1) = 1;
izlaz(2, klasa == 2) = 1;
izlaz(3, klasa == 3) = 1;

ulaz = ob;

%% Podela na trening i test skup
ind = randperm(N);

indTrening = ind(1 : 450);
indTest = ind(451 : 500);

indTrening2 = ind(501 : 950);
indTest2 = ind(951 : 1000);
indTrening3 = ind(1001 : 1450);
indTest3 = ind(1451 : 1500);




ulazTrening = ulaz(:, indTrening);
izlazTrening = izlaz(:, indTrening);

ulazTrening2 = ulaz(:, indTrening2);
izlazTrening2 = izlaz(:, indTrening2);


ulazTrening3 = ulaz(:, indTrening3);
izlazTrening3 = izlaz(:, indTrening3);



ulazTest = ulaz(:, indTest);
izlazTest = izlaz(:, indTest);

ulazTest2 = ulaz(:, indTest2);
izlazTest2 = izlaz(:, indTest2);

ulazTest3 = ulaz(:, indTest3);
izlazTest3 = izlaz(:, indTest3);
%% Kreiranje NM
arhitektura = [5 5 5];
net = patternnet(arhitektura);

for i = 1 : length(arhitektura)
    net.layers{i}.transferFcn = 'tansig';
end
%net.layers{length(arhitektura) + 1}.transferFcn = 'softmax';

net.performFcn = 'crossentropy';

% net.performParam.regularization = 0.2;

% net.divideFcn = '';
net.divideFcn = '';
%net.divideParam.trainRatio = 0.9;
%net.divideParam.valRatio = 0.1;
%net.divideParam.testRatio = 0;

net.trainParam.epochs = 120;
net.trainParam.goal = 1e-4;
net.trainParam.min_grad = 1e-5;
net.trainParam.max_fail = 20;





arhitektura1 = [2 2 2];
net1 = patternnet(arhitektura1);

for i = 1 : length(arhitektura1)
    net1.layers{i}.transferFcn = 'tansig';
end
%net1.layers{length(arhitektura1) + 1}.transferFcn = 'softmax';

net1.performFcn = 'crossentropy';

% net.performParam.regularization = 0.2;

% net.divideFcn = '';
net1.divideFcn = '';
%net1.divideParam.trainRatio = 0.9;
%net1.divideParam.valRatio = 0.1;
%net1.divideParam.testRatio = 0;

net1.trainParam.epochs = 120;
net1.trainParam.goal = 1e-4;
net1.trainParam.min_grad = 1e-5;
net1.trainParam.max_fail = 20;






arhitektura2 = [100 100 100 100];
net2 = patternnet(arhitektura2);

for i = 1 : length(arhitektura2)
    net2.layers{i}.transferFcn = 'tansig';
end
%net2.layers{length(arhitektura2) + 1}.transferFcn = 'softmax';

net2.performFcn = 'crossentropy';

% net.performParam.regularization = 0.2;

% net.divideFcn = '';
net2.divideFcn = '';
%net2.divideParam.trainRatio = 0.9;
%net2.divideParam.valRatio = 0.1;
%net2.divideParam.testRatio = 0;

net2.trainParam.epochs = 120;
net2.trainParam.goal = 1e-4;
net2.trainParam.min_grad = 1e-5;
net2.trainParam.max_fail = 20;

%% Treniranje NM
[net, tr] = train(net, ulazTrening, izlazTrening);
[net1, tr2] = train(net1, ulazTrening2, izlazTrening2);
[net2, tr3] = train(net2, ulazTrening3, izlazTrening3);

%% Performanse NM
predTest = net(ulazTest);

%%%%%% izvrsiti predikciju NM nad test podacima
izlazPredTest1 = sim(net, ulazTest);
izlazPredTest2 = sim(net1, ulazTest2);
izlazPredTest3 = sim(net2, ulazTest3);

%%%%%% Prikazati matricu konfuzije za trening skup
figure 
plotconfusion(izlazTest, izlazPredTest1, 'TestOpt');
figure 
plotconfusion(izlazTest2, izlazPredTest2, 'TestUnder');
figure 
plotconfusion(izlazTest3, izlazPredTest3, 'TestOver');

%%%%%% izvrsiti predikciju NM nad trening podacima
izlazPredTrening1 = sim(net,ulazTrening);
izlazPredTrening2 = sim(net1,ulazTrening2);
izlazPredTrening3 = sim(net2,ulazTrening3);


%%%%%% Prikazati matricu konfuzije za test skup
figure 
plotconfusion(izlazTrening, izlazPredTrening1, 'TreningOpt');
figure
plotconfusion(izlazTrening2, izlazPredTrening2, 'TreningUnder');
figure
plotconfusion(izlazTrening3, izlazPredTrening3, 'TreningOver');

[c, cm] = confusion(izlazTest, predTest);
cm = cm';

% K1 kao klasa od interesa
P = cm(1, 1)/sum(cm(1, :));
R = cm(1, 1)/sum(cm(:, 1));
F1 = 2*P*R/(P+R);

disp(P)
disp(R)
disp(F1)

%% Granica odlucivanja
Ntest = 500;
x1 = repmat(linspace(-6, 6, Ntest), 1, Ntest);
x2 = repelem(linspace(-6, 6, Ntest), Ntest);
ulazGO = [x1; x2];

predGO = net(ulazGO);
[vr, klasaGO] = max(predGO);

K1go = ulazGO(:, predGO(1, :) >= 0.9);
K2go = ulazGO(:, predGO(2, :) >= 0.9);
K3go = ulazGO(:, predGO(3, :) >= 0.9);

figure, hold all




plot(K1go(1, :), K1go(2, :), '.')
plot(K2go(1, :), K2go(2, :), '.')
plot(K3go(1, :), K3go(2, :), '.')
plot(K1(1, :), K1(2, :), 'bo')
plot(K2(1, :), K2(2, :), 'r*')
plot(K3(1, :), K3(2, :), 'yd')


x1 = repmat(linspace(-6, 6, Ntest), 1, Ntest);
x2 = repelem(linspace(-6, 6, Ntest), Ntest);
ulazGO = [x1; x2];

predGO = net1(ulazGO);
[vr2, klasaGO] = max(predGO);

K1go = ulazGO(:, predGO(1, :) >= 0.9);
K2go = ulazGO(:, predGO(2, :) >= 0.9);
K3go = ulazGO(:, predGO(3, :) >= 0.9);

figure, hold all


plot(K1go(1, :), K1go(2, :), '.')
plot(K2go(1, :), K2go(2, :), '.')
plot(K3go(1, :), K3go(2, :), '.')
plot(K1(1, :), K1(2, :), 'bo')
plot(K2(1, :), K2(2, :), 'r*')
plot(K3(1, :), K3(2, :), 'yd')



x1 = repmat(linspace(-6, 6, Ntest), 1, Ntest);
x2 = repelem(linspace(-6, 6, Ntest), Ntest);
ulazGO = [x1; x2];

predGO = net2(ulazGO);
[vr2, klasaGO] = max(predGO);

K1go = ulazGO(:, predGO(1, :) >= 0.9);
K2go = ulazGO(:, predGO(2, :) >= 0.9);
K3go = ulazGO(:, predGO(3, :) >= 0.9);

figure, hold all


plot(K1go(1, :), K1go(2, :), '.')
plot(K2go(1, :), K2go(2, :), '.')
plot(K3go(1, :), K3go(2, :), '.')
plot(K1(1, :), K1(2, :), 'bo')
plot(K2(1, :), K2(2, :), 'r*')
plot(K3(1, :), K3(2, :), 'yd')

figure, hold all
plotperform(tr)
figure, hold all
plotperform(tr2)
figure, hold all
plotperform(tr3)
