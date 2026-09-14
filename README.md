# Neural Networks

## Architecutre

| Neural netowrk | Digital neuron |
|-|-|
| <img src="img/cartoons/small_nn-1.png"> | <img src="img/cartoons/neuron_model-1.png">
<details>

```
pdflatex \
    -output-directory=img/cartoons \
    img/cartoons/small_nn.tex

pdftoppm \
    -png \
    -r 150 \
    img/cartoons/small_nn.pdf \
    img/cartoons/small_nn

pdflatex \
    -output-directory=img/cartoons \
    img/cartoons/neuron_model.tex

pdftoppm \
    -png \
    -r 150 \
    img/cartoons/neuron_model.pdf \
    img/cartoons/neuron_model
```

</details>

- Digital neuron
  - Inputs ($x_1 \dots x_m$): Numerical feature signals fed directly into the
    neuron from external data or upstream neurons.
  - Weights ($w_{k1} \dots w_{km}$): Adjustable numerical coefficients assigned
    to every incoming connection that dictate the strength and influence of
    each input signal. During training, the network updates these parameters to
    learn which features matter most for accurate predictions.
  - Aggregation ($\sum$): Each input value is multiplied by its corresponding
    weight and summed together. This collects all incoming weighted signals
    into a single scalar value.
  - Bias: A learnable constant value added to the aggregated weighted sum. It
    acts as an offset, allowing the activation function to shift left or right
    along the axis so the neuron can fit data patterns that do not pass through
    the origin.
  - Activation funciton ($\varphi$): Mathematical functions applied to the
    aggregated sum and bias to introduce non-linearity, which is essential for
    learning complex, real-world relationships. Common
    choices include:
      - ReLU: Zeroes out negative inputs to introduce non-linearity efficiently
        and encourage sparsity.
        <img src="img/act_relu.png" style="height: 1in;">
        <details>

        ```bash
        python3 -c "
            import numpy as np
            x = np.linspace(-6, 6, 200)
            y = np.maximum(x, 0)
            for xi, yi in zip(x, y): print(xi, yi)
        " | python3 src/plot_line.py \
            -o img/act_relu.png \
            --width 1 \
            --height 1 --line_style "-"
        ```

        </details>
      - Sigmoid: Squashes values into a range between $0$ and $1$, making it
        ideal for binary probabilities.
        <img src="img/act_sigmoid.png" style="height: 1in;">
        <details>

        ```bash
        python3 -c "
            import numpy as np
            x = np.linspace(-6, 6, 200)
            y = 1/(1+np.exp(-x))
            for xi, yi in zip(x, y): print(xi, yi)
        " \
        | python3 src/plot_line.py \
            -o img/act_sigmoid.png \
            --width 1 \
            --height 1 \
            --line_style "-"
        ```

        </details>

      - Tanh: Maps values between $-1$ and $1$, providing zero-centered outputs
        for smoother training.
        <img src="img/act_tanh.png" style="height: 1in;">
        <details>

        ```bash
        python3 -c "
            import numpy as np
            x = np.linspace(-6, 6, 200)
            y = np.tanh(x)
            for xi, yi in zip(x, y): print(xi, yi)
        " \
        | python3 src/plot_line.py \
            -o img/act_tanh.png \
            --width 1 \
            --height 1 \
            --line_style "-"
        ```

        </details>

  - Output: The final scalar output signal produced by the neuron after
    activation. This value serves as the single prediction for binary models or
    is broadcast forward as an input to downstream digital neurons in a
    network.




## Gernate interval data file

A NN can fit a curve to almost any dataset
We are going to use a complete sytetic dataset 
Goign to try and find the start and end of an interval
This could be the effective dosage range for some treament

<img src="img/interval.data.png" style="height: 2in;">

<details>

```bash
python src/make_interval_dataset.py \
    --out out/interval.data.tsv
wrote out/interval.data.tsv: 200 points, 32.0% positive, baseline accuracy = 0.680

tail -n +2 out/interval.data.tsv \
| python src/plot_line.py \
    -o img/interval.data.png \
    -x "x" \
    -y "label" \
    --point_size 4 \
    --markeredgecolor "tab:blue" \
    --markerfacecolor "tab:blue"
```

</details>


## Train

```bash
python src/train_interval_nn.py \
    --data out/interval.data.tsv \
    --out_prefix out/interval
```
