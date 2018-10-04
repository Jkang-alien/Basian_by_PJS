library(rstan)

x <- c()
s <- c()
b <- c()
for (i in 1:20){
  v <- sample(seq(100, 800, by = 100), 1, prob=rev(seq(0.1, 0.8, by = 0.1)))
  s <- append(s, v)
  b <- sample((v-20):(v+20), 10, rep = TRUE)
  x <- append(x, b)
}
 
l =c()
for (i in 1:length(s)) {
  lead_location <- rep(i, 10)
  l <- append(l, lead_location)
}

Nsub <- length(s)
f <- sample(seq(0.1, 0.2, by = 0.01), length(x), replace = TRUE)
r <- (log2(x/mean(x)))

mydata <- list(r = r, f = f, N=length(r), s = l, Nsub = Nsub, psy = mean(x)/100)

code <- '
data {
  int N;
  real r[N];
  real f[N];
  int<lower=1> Nsub;
  int<lower=1> s[N];
  real psy;
}
parameters {
  real<lower=0.1, upper=20> cn[Nsub];
  real<lower=0, upper=5> m[Nsub];
  real<lower=0> sigma_cn[Nsub];
  real<lower=0> sigma_m[Nsub];
  real<lower=0, upper=1.0> P;
}
model {
  for(i in 1:N){
  r[i] ~ normal(log2((P*cn[s[i]] + 2*(1-P))/(P*psy + 2*(1-P))),sigma_cn[s[i]]);
  f[i] ~ normal((P*m[s[i]]+1-P)/(P*cn[s[i]]+2*(1-P)), sigma_m[s[i]]);
  }
}
'

fit <- stan(model_code = code, data = mydata, iter = 1000, 
            chains = 2, control = list(adapt_delta = 0.99,
                                       max_treedepth = 15))

plot(fit, pars = 'cn')

traceplot(fit, pars = 'cn')

stan_diag(fit)
post <- extract(fit)

hist(post$P,
     main = paste("Tumor purity"),
     ylab = '')

