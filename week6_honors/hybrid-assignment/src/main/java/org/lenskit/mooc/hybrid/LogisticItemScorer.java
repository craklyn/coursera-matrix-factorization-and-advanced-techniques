package org.lenskit.mooc.hybrid;

import it.unimi.dsi.fastutil.longs.LongSet;
import org.apache.commons.math3.linear.RealVector;
import org.lenskit.api.ItemScorer;
import org.lenskit.api.Result;
import org.lenskit.api.ResultMap;
import org.lenskit.basic.AbstractItemScorer;
import org.lenskit.bias.BiasModel;
import org.lenskit.bias.UserBiasModel;
import org.lenskit.data.ratings.RatingSummary;
import org.lenskit.results.Results;
import org.lenskit.util.collections.LongUtils;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Item scorer that does a logistic blend of a subsidiary item scorer and popularity.  It tries to predict
 * whether a user has rated a particular item.
 */
public class LogisticItemScorer extends AbstractItemScorer {
    private final LogisticModel logisticModel;
    private final BiasModel biasModel;
    private final RecommenderList recommenders;
    private final RatingSummary ratingSummary;

    @Inject
    public LogisticItemScorer(LogisticModel model, UserBiasModel bias, RecommenderList recs, RatingSummary rs) {
        logisticModel = model;
        biasModel = bias;
        recommenders = recs;
        ratingSummary = rs;
    }

    @Nonnull
    @Override
    public ResultMap scoreWithDetails(long user, @Nonnull Collection<Long> items) {

        // My code:
        // Implement item scorer
        // throw new UnsupportedOperationException("item scorer not implemented");

        LongSet itemSet = LongUtils.asLongSet(items);
        List<Result> results = new ArrayList<>();

        double model_int = logisticModel.getIntercept();
        RealVector model_coef = logisticModel.getCoefficients();

        for(Long item : itemSet) {
            double baseline_bias = biasModel.getIntercept() + biasModel.getItemBias(item) +
                    biasModel.getUserBias(user);
            double log_popularity = Math.log(ratingSummary.getItemRatingCount(item));

            double linear_out = model_int + model_coef.getEntry(0) * baseline_bias +
                    model_coef.getEntry(1) * log_popularity;

            for (int i = 2; i < model_coef.getDimension(); i++) {
                if (recommenders.getItemScorers().get(i - 2).score(user, item) != null) {
                    double model_score = recommenders.getItemScorers().get(i - 2).score(user, item).getScore() - baseline_bias;
                    linear_out += model_coef.getEntry(i) * model_score;
                }
            }

            double pred = LogisticModel.sigmoid(linear_out);
            Result r = Results.create(item, pred);
            // Store the results in 'results'
            results.add(r);
        }

        return Results.newResultMap(results);

    }
}
