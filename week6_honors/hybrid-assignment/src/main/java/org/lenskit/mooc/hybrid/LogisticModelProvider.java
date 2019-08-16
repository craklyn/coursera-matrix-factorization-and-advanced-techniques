package org.lenskit.mooc.hybrid;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.lenskit.api.ItemScorer;
import org.lenskit.api.Result;
import org.lenskit.bias.BiasModel;
import org.lenskit.bias.UserBiasModel;
import org.lenskit.data.ratings.Rating;
import org.lenskit.data.ratings.RatingSummary;
import org.lenskit.inject.Transient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.inject.Inject;
import javax.inject.Provider;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

class Triplet {
    int i;
    long j;
    long k;

    Triplet(int i,long j, long k) {
        this.i = i;
        this.j = j;
        this.k = k;
    }

    public boolean equals(Object other)
    {
        if (other == null) {
            return false;
        }

        if (this.getClass() != other.getClass()) {
            return false;
        }

        Triplet other_quartet = (Triplet) other;

        return this.i == other_quartet.i &&
                this.j == other_quartet.j &&
                this.k == other_quartet.k;
    }

    public int hashCode() {
        return (new Integer(this.i)).hashCode() +
                (new Long(this.j)).hashCode() +
                (new Long(this.k)).hashCode();
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
    private final HashMap<Triplet, Double> getScoreCache;
    private int cache_lookup_count;
    private int cache_not_found_count;

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
        getScoreCache = new HashMap<>();
        cache_lookup_count = 0;
        cache_not_found_count = 0;

    }

    public double getScore(int recommender_index, long user_id, long item_id) {
        Triplet tuple_values = new Triplet(recommender_index, user_id, item_id);
        if (getScoreCache.containsKey(tuple_values )) {
            cache_lookup_count += 1;
            return getScoreCache.get(tuple_values);
        }

        cache_not_found_count += 1;

        if (recommenders.getItemScorers().get(recommender_index).score(user_id, item_id) != null) {
            double bias_ui = baseline.getIntercept() + baseline.getItemBias(item_id) +
                    baseline.getUserBias(user_id);
            double model_score = recommenders.getItemScorers().get(recommender_index).score(user_id, item_id).getScore()
                    - bias_ui;

            getScoreCache.put(tuple_values, model_score);
            return model_score;
        }
        else {

            getScoreCache.put(tuple_values, 0.);
            return 0.;
        }
    }

    public double getScore_noCache(int recommender_index, long user_id, long item_id) {
        if (recommenders.getItemScorers().get(recommender_index).score(user_id, item_id) != null) {
            double bias_ui = baseline.getIntercept() + baseline.getItemBias(item_id) +
                    baseline.getUserBias(user_id);
            double model_score = recommenders.getItemScorers().get(recommender_index).score(user_id, item_id).getScore()
                    - bias_ui;
            return model_score;
        }
        else {
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

                RealVector X = new ArrayRealVector(parameterCount);

                double bias_ui = baseline.getIntercept() + baseline.getItemBias(item_id) +
                        baseline.getUserBias(user_id);
                X.setEntry(0, bias_ui);

                double log_popularity = Math.log(ratingSummary.getItemRatingCount(item_id));
                X.setEntry(1, log_popularity);


                for (int j = 2; j < params.length; j++) {
                    X.setEntry(j, getScore(j-2, user_id, item_id));
                }

                double activation_out = current.evaluate(-y_ui, X);

                intercept += LEARNING_RATE * y_ui * activation_out;
                for (int j = 0; j < parameterCount; j++){
                    params[j] += LEARNING_RATE * y_ui * X.getEntry(j) * activation_out;
                }

                current = LogisticModel.create(intercept, params);
            }

            /*
            System.out.print(intercept + " ");
            for(int j=0; j<params.length; j++){
                System.out.print(params[j] + " ");
            }
            System.out.println("");

             */
        }

        /*
        System.out.println("Number of cache not founds / new cache items: " + cache_not_found_count);
        System.out.println("Number of cache lookups: " + cache_lookup_count);
        System.out.println("getScoreCache size: " + getScoreCache.size());
         */

        return current;
    }

}
