package org.lenskit.mooc.hybrid;

import com.google.common.base.Preconditions;
import org.lenskit.api.ItemScorer;
import org.lenskit.api.Result;
import org.lenskit.api.ResultMap;
import org.lenskit.basic.AbstractItemScorer;
import org.lenskit.bias.BiasModel;
import org.lenskit.results.Results;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Item scorer that computes a linear blend of two scorers' scores.
 *
 * <p>This scorer takes two underlying scorers and blends their scores.
 */
public class LinearBlendItemScorer extends AbstractItemScorer {
    private final BiasModel biasModel;
    private final ItemScorer leftScorer, rightScorer;
    private final double blendWeight;

    /**
     * Construct a popularity-blending item scorer.
     *
     * @param bias The baseline bias model to use.
     * @param left The first item scorer to use.
     * @param right The second item scorer to use.
     * @param weight The weight to give popularity when ranking.
     */
    @Inject
    public LinearBlendItemScorer(BiasModel bias,
                                 @Left ItemScorer left,
                                 @Right ItemScorer right,
                                 @BlendWeight double weight) {
        Preconditions.checkArgument(weight >= 0 && weight <= 1, "weight out of range");
        biasModel = bias;
        leftScorer = left;
        rightScorer = right;
        blendWeight = weight;
    }

    @Nonnull
    @Override
    public ResultMap scoreWithDetails(long user, @Nonnull Collection<Long> items) {
        List<Result> results = new ArrayList<>();

        // My code:
        // Compute hybrid scores
        for (Long item : items) {

            double baseline = biasModel.getIntercept() + biasModel.getItemBias(item) + biasModel.getUserBias(user);
            double leftOffset, rightOffset;

            if (leftScorer.score(user, item) != null) {
                leftOffset = leftScorer.score(user, item).getScore() - baseline;
            }
            else {
                leftOffset = 0;
            }

            if (rightScorer.score(user, item) != null) {
                rightOffset = rightScorer.score(user, item).getScore() - baseline;
            }
            else {
                rightOffset = 0;
            }

            double prediction = baseline + (1 - blendWeight) * leftOffset + blendWeight * rightOffset;

            Result r = Results.create(item, prediction);
            results.add(r);
        }

        return Results.newResultMap(results);
    }
}
