function [trainingDataA, trainingDataB, verifyDataA, verifyDataB]=splitData(dataA, dataB,trainingRatioA, trainingRatioB)

[~, lenA] = size(dataA);
[~, lenB] = size(dataB);

randomA = dataA(:, randperm(size(dataA, 2))); 
for i=1:lenA
    if i <= lenA * trainingRatioA
        trainingDataA(:, i) = randomA(:, i);
    else 
        verifyDataA(:, i) = randomA(:,i);
    end
end

randomB = dataB(:, randperm(size(dataB, 2))); 
for i=1:lenB
    if i <= lenB * trainingRatioB
        trainingDataB(:,i) = randomB(:, i);
    else 
        verifyDataB(:,i) = randomB(:,i);
    end
end

if trainingRatioB == 1
    verifyDataB = [];
end
if trainingRatioA == 1
    verifyDataA = [];
end
