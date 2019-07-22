(ns mnist-classifier.cost
  (:use [mnist-classifier.util])
  (:require [clojure.core.matrix :as m]
            [clojure.core.reducers :as r]))

(defn cost
  "Calculates regularised cost of network."
  [X y lambda weights]
  (let [m (m/row-count X)
        [theta1 theta2] weights
        ; Add bias term (ones) to input layer
        a1 (m/join-along 1 (one-matrix m 1) X)
        ; Multiply first layer by first set of weights
        z2 (m/mmul a1 (m/transpose theta1))
        ; Compute activation and add bias term
        a2 (m/join-along 1 (one-matrix m 1) (m/logistic z2))
        ; Multiply second layer by second set of weights
        z3 (m/mmul a2 (m/transpose theta2))
        ; Compute final activation (this is h)
        a3 (m/logistic z3)
        ; Turn each integer i in vector y into a zero-filled vector with
        ; element i set to 1 so that 3 becomes [0 0 0 1 0 0 0 0 0 0 0 ]
        Y (map #(m/set-row (m/zero-vector 10) % 1) y)

        ; Compute regularisation component of cost.
        ; (Ignore bias term in regularisation.)
        theta1-no-bias (m/square (map #(drop 1 %) theta1))
        theta2-no-bias (m/square (map #(drop 1 %) theta2))

        reg (* (+ (m/esum (m/square theta1-no-bias))
                  (m/esum (m/square theta2-no-bias)))
               (/ lambda (* 2 m)))

        ; Cost without regularisation
        cost (->
               (m/negate Y)
               (m/emul ,,, (m/log a3))
               (m/sub ,,, (->
                            (m/sub 1 Y)
                            (m/emul ,,, (m/log (m/sub 1 a3))))))]

    ; This is J
    (+ reg (/ (m/esum cost) m))))

(defn back-prop
  "Performs back propagation on one training example.
  `t` is the target vector for training example `x`.
  Returns two matrices of gradients stored in a single vector."
  [x t weights gradients]
  (let [[theta1 theta2] weights
        [theta1-grad theta2-grad] gradients

        ; Feed forward
        a1 (m/join [1] x)
        z2 (m/mmul a1 (m/transpose theta1))
        a2 (m/join [1] (m/logistic z2))
        z3 (m/mmul a2 (m/transpose theta2))
        a3 (m/logistic z3)

        ; Error terms for this example
        delta-3 (m/sub a3 t)

        delta-2 (->>
                  (m/square (map #(drop 1 %) theta2))
                  (m/mmul ,,, delta-3)
                  (m/emul ,,, (sigmoid-prime z2)))

        ; Accumulate gradient
        theta2-grad (->
                      (m/transpose [delta-3])
                      (m/mmul [a2] ,,,)
                      (m/add theta2-grad) ,,,)

        theta1-grad (->
                      (m/transpose [delta-2])
                      (m/mmul [a1] ,,,)
                      (m/add theta1-grad) ,,,)]

    [theta1-grad theta2-grad]))

(defn gradient
  "Calculates partial derivatives of cost function."
  [X y lambda weights]
  (let [m (m/row-count X)
        ; Extract thetas from weights vector
        [theta1 theta2] weights

        input-layer (count (first X))
        hidden-layer (m/row-count theta1)
        output-layer (m/row-count theta2)

        ; Partial derivatives
        theta1-grad (m/zero-matrix hidden-layer (inc input-layer))
        theta2-grad (m/zero-matrix output-layer (inc hidden-layer))

        Y (class-vector y)]

    (def gradients
      (reduce
        (fn [g [x t]]
          (back-prop x t [theta1 theta2] g))
        [theta1-grad theta2-grad]
        (map vector X Y)))

    ; The two procedures below for back prop work as intented. Wrote them to
    ; check if the functional map/reduce implementation above also works.

    ; Back prop with loop
    ;(loop [i 0]
    ;  (when (< i m)
    ;    (let [x (m/get-row X i)
    ;         t (m/get-row Y i)]
    ;      (def gradients (back-prop x t [theta1 theta2] gradients))
    ;    (recur (inc i)))))

    ; Back prop with seq
    ;(doseq [i (range m)
    ;        :let [x (m/get-row X i)
    ;              t (m/get-row Y i)]]
    ;  (def gradients (back-prop x t [theta1 theta2] gradients)))

    (let [[theta1-grad theta2-grad] gradients

          ; Obtain gradient (unregularized)
          theta1-grad (m/div theta1-grad m)
          theta2-grad (m/div theta2-grad m)

          ; Add regularization (ignore bias term)
          theta1-reg (->
                       (drop 1 theta1-grad)
                       (m/add ,,, (/ lambda m))
                       (m/emul ,,, (drop 1 theta1)))

          theta1-grad (m/join (take 1 theta1-grad) theta1-reg)

          theta2-reg (->
                       (drop 1 theta2-grad)
                       (m/add ,,, (/ lambda m))
                       (m/emul ,,, (drop 1 theta2)))

          theta2-grad (m/join (take 1 theta2-grad) theta2-reg)]

      [theta1-grad theta2-grad])))

(defn cost-function
  "Generates a cost function which takes network weights as a single parameter.

  The function returns both the cost of network and the partial derivatives of
  the cost function for the given feature-target set (`X` and `y`) and lambda.

  The parameter is a combined vector of layer weights as required by the
  minimiser. This essentially creates a shorthand version of the full cost
  function."
  [X y lambda]
  (defn cf [weights]
    [(cost X y lambda weights) (gradient X y lambda weights)]))

