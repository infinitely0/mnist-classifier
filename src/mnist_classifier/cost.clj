(ns mnist-classifier.cost
  (:use [mnist-classifier.util])
  (:require [clojure.core.matrix :as m]
            [clojure.core.reducers :as r])
  (:import org.ejml.simple.SimpleMatrix))

(defn cost
  "Calculates regularised cost of network."
  [X Y lambda weights]
  (let [m (.numRows X)
        [theta1 theta2] weights

        ; Add bias term (ones) to input layer
        a1 (.combine (one-sm-matrix m 1) 0 SimpleMatrix/END X)

        ; Multiply first layer by first set of weights
        z2 (.mult a1 (.transpose theta1))

        ; Compute activation and add bias term
        a2 (.combine (one-sm-matrix m 1) 0 SimpleMatrix/END (sigmoid z2))

        ; Multiply second layer by second set of weights
        z3 (.mult a2 (.transpose theta2))

        ; Compute final activation (this is h)
        a3 (sigmoid z3)

        ; Compute regularisation component of cost.
        ; (Ignore bias term in regularisation.)
        reg (* (+ (.elementSum (.elementPower (drop-first-column theta1) 2.0))
                  (.elementSum (.elementPower (drop-first-column theta2) 2.0)))
               (/ lambda (* 2 m)))

        ; Cost without regularisation
        cost (->
               (.negative Y)
               (.elementMult ,,, (.elementLog a3))
               (.minus ,,, (->
                             (.minus
                               (one-sm-matrix (.numRows Y)
                                              (.numCols Y))
                               Y)
                             (.elementMult ,,, (.elementLog
                                                 (.minus
                                                   (one-sm-matrix (.numRows a3)
                                                                  (.numCols a3))
                                                   a3))))))

        ]

    ; Cost with reg (J)
    (+ reg (/ (.elementSum cost) m))))

(defn back-prop
  "Performs back propagation on one training example.
  `y` is the target vector for training example `x`.
  Returns two matrices of gradients stored in a single vector."
  [x y weights gradients]
  (let [[theta1 theta2] weights
        [theta1-grad theta2-grad] gradients

        ; Feed forward
        a1 (.combine (one-sm-matrix 1 1) 0 SimpleMatrix/END x)
        z2 (.mult a1 (.transpose theta1))
        a2 (.combine (one-sm-matrix 1 1) 0 SimpleMatrix/END (sigmoid z2))
        z3 (.mult a2 (.transpose theta2))
        a3 (sigmoid z3)

        ; Error terms for this example
        delta-3 (.minus a3 y)

        delta-2 (->>
                  (drop-first-column theta2)
                  (.mult delta-3 ,,,)
                  (.elementMult (sigmoid-prime z2) ,,,))

        ; Accumulate gradient
        theta2-grad-acc (->
                          (.transpose delta-3)
                          (.mult ,,, a2)
                          (.plus ,,, theta2-grad))

        theta1-grad-acc (->
                          (.transpose delta-2)
                          (.mult a1 ,,,)
                          (.plus theta1-grad ,,,))]

    [theta1-grad-acc theta2-grad-acc]))

(defn gradient
  "Calculates partial derivatives of cost function."
  [X Y lambda weights]
  (let [m (.numRows X)

        [theta1 theta2] weights

        input-layer (.numCols X)
        hidden-layer (.numRows theta1)
        output-layer (.numRows theta2)

        ; Partial derivatives
        theta1-grad (SimpleMatrix. hidden-layer (inc input-layer))
        theta2-grad (SimpleMatrix. output-layer (inc hidden-layer))]

    (loop [i 0]
      (when (< i m)
        (let [x (.extractVector X true i)
              t (.extractVector Y true i)]
          (def gradients
            (back-prop x t
                       [theta1 theta2]
                       [theta1-grad theta2-grad]))
        (recur (inc i)))))

    ; Not sure how to use reduce with the Java library
    ;(reduce
    ;  (fn [g [x t]]
    ;    (back-prop x t [theta1-sm theta2-sm] g))
    ;  [theta1-grad-sm theta2-grad-sm]
    ;  (map vector X Y))))

    (let [[theta1-grad theta2-grad] gradients

          ; Obtain gradient (unregularized)
          theta1-grad (.divide theta1-grad m)
          theta2-grad (.divide theta2-grad m)

          ; Add regularization (ignore bias term)
          theta1-reg (->
                          (drop-first-column theta1-grad)
                          (.plus ,,, (double (/ lambda m)))
                          (.elementMult ,,, (drop-first-column theta1)))

          theta1-grad (.combine (take-first-column theta1-grad)
                                0 SimpleMatrix/END
                                theta1-reg)

          theta2-reg (->
                          (drop-first-column theta2-grad)
                          (.plus ,,, (double (/ lambda m)))
                          (.elementMult ,,, (drop-first-column theta2)))

          theta2-grad (.combine (take-first-column theta2-grad)
                                   0 SimpleMatrix/END
                                   theta2-reg)]

      [theta1-grad theta2-grad])))

(defn cost-function
  "Generates a cost function which takes network weights as a single parameter.

  The function returns both the cost of network and the partial derivatives of
  the cost function for the given feature-target set (`X` and `Y`) and lambda.

  The parameter is a combined vector of layer weights as required by the
  minimiser. This essentially creates a shorthand version of the full cost
  function."
  [X Y lambda]
  (defn cf [weights]
    [(cost X Y lambda weights) (gradient X Y lambda weights)]))

