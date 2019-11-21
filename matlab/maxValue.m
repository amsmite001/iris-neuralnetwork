function [maxIndex] = maxValue(neural)
maxA=neural(1);
maxIndex=1;
    for i=2:numel(neural)
        if neural(i) > maxA
            maxA=neural(i);
            maxIndex=i;
        end
    end
end