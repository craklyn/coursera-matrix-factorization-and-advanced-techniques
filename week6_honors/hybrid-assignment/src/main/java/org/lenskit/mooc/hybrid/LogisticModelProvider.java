package org.lenskit.mooc.hybrid;

import org.lenskit.api.ItemScorer;
import org.lenskit.api.Result;
import org.lenskit.bias.BiasModel;
import org.lenskit.bias.UserBiasModel;
import org.lenskit.data.ratings.Rating;
import org.lenskit.data.ratings.RatingSummary;
import org.lenskit.inject.Transient;
import org.lenskit.util.ProgressLogger;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.inject.Inject;
import javax.inject.Provider;
import java.util.*;
import java.util.concurrent.TimeUnit;

class Quartet {
    int i;
    long j;
    long k;
    double l;

    Quartet(int i,long j, long k, double l) {
        this.i = i;
        this.j = j;
        this.k = k;
        this.l = l;
    }

    public boolean equals(Object other)
    {
        if (other == null) {
            return false;
        }

        if (this.getClass() != other.getClass()) {
            return false;
        }

        Quartet other_quartet = (Quartet) other;

        return this.i == other_quartet.i &&
                this.j == other_quartet.j &&
                this.k == other_quartet.k &&
                this.l == other_quartet.l;
    }

    public int hashCode() {
        return (new Integer(this.i)).hashCode() +
                (new Long(this.j)).hashCode() +
                (new Long(this.k)).hashCode() +
                (new Double(this.l)).hashCode();
    }

}

/**
 * Trainer that builds logistic models.
 */
public class LogisticModelProvider implements Provider<LogisticModel> {
    private static final Logger logger = LoggerFactory.getLogger(LogisticModelProvider.class);
    private static final double LEARNING_RATE = 0.00005;
    private static final int ITERATION_COUNT = 100;

    private final LogisticTrainingSplit dataSplit;
    private final BiasModel baseline;
    private final RecommenderList recommenders;
    private final RatingSummary ratingSummary;
    private final int parameterCount;
    private final Random random;
    private final HashMap<Quartet, Double> getScoreCache;

    @Inject
    public LogisticModelProvider(@Transient LogisticTrainingSplit split,
                                 @Transient UserBiasModel bias,
                                 @Transient RecommenderList recs,
                                 @Transient RatingSummary rs,
                                 @Transient Random rng) {
        dataSplit = split;
        baseline = bias;
        recommenders = recs;
        ratingSummary = rs;
        parameterCount = 1 + recommenders.getRecommenderCount() + 1;
        random = rng;
        getScoreCache = new HashMap<Quartet, Double>();

    }

    public double getScore(int recommender_index, long user_id, long item_id, double baseline_bias) {
        Quartet tuple_values = new Quartet(recommender_index, user_id, item_id, baseline_bias);
        if (getScoreCache.containsKey(tuple_values )) {
//            System.out.println("Found key in cache.");
            return getScoreCache.get(tuple_values);
        }

//        System.out.println("Key not found in cache.");

        if (recommenders.getItemScorers().get(recommender_index).score(user_id, item_id) != null) {
            double model_score = recommenders.getItemScorers().get(recommender_index).score(user_id, item_id).getScore()
                    - baseline_bias;

            getScoreCache.put(tuple_values, model_score);
            return model_score;
        }
        else {

            getScoreCache.put(tuple_values, 0.);
            return 0.;
        }
    }


    @Override
    public LogisticModel get() {
        List<ItemScorer> scorers = recommenders.getItemScorers();
        double intercept = 0;
        double[] params = new double[parameterCount];

        LogisticModel current = LogisticModel.create(intercept, params);

        // My code:
        // Implement model training
        List<Rating> tuneRatings = dataSplit.getTuneRatings();

        for (int i = 0; i < ITERATION_COUNT; i++) {
            Collections.shuffle(tuneRatings);

            for (Rating rating : tuneRatings) {
                // x1 is bias (from a BiasModel)
                // x2 will be popularity (ln |R_i|)
                // x3, x4, .. are scores from other item scorers
                long user_id = rating.getUserId();
                long item_id = rating.getItemId();

                double y_ui = rating.getValue();
                double baseline_bias = baseline.getIntercept() + baseline.getItemBias(item_id) +
                        baseline.getUserBias(user_id);
                double log_popularity = Math.log(ratingSummary.getItemRatingCount(item_id));

                double linear_out = intercept + params[0] * baseline_bias + params[1] * log_popularity;
                for (int j = 2; j < params.length; j++) {
                    linear_out += getScore(j-2, user_id, item_id, baseline_bias);
                }

                double activation_out = LogisticModel.sigmoid(-y_ui * linear_out);

                intercept += LEARNING_RATE * y_ui * activation_out;
                params[0] += LEARNING_RATE * y_ui * baseline_bias;
                params[1] += LEARNING_RATE * y_ui * log_popularity;
                for (int j = 2; j < params.length; j++) {
                    params[j] += LEARNING_RATE * y_ui * getScore(j-2, user_id, item_id, baseline_bias) * activation_out;
                }
            }

            current = LogisticModel.create(intercept, params);
        }

        return current;
    }

}
