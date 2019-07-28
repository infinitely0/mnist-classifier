(ns mnist-classifier.core
  (:use [mnist-classifier.cost]
        [mnist-classifier.train]
        [mnist-classifier.util])
  (:require [clojure.core.matrix :as m]))

(def weights (train))

(def X-test (m/matrix (extract-features (read-csv "resources/sample-test.csv"))))
(def y-test (m/matrix (extract-labels (read-csv "resources/sample-test.csv"))))

(def accuracy
  (let [m (count y-test)
        predictions (predict-all X-test weights)
        results (map = predictions y-test)
        score (map #(if % 1 0) results)
        total (m/non-zero-count score)]
    (float (/ total m))))

(defn -main
  [& args]
  (println "Accuracy: " accuracy))

