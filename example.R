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
f <- sample(seq(0.2, 0.5, by = 0.1), length(x),
            prob = c(0.1, 0.1, 0.1, 0.9), replace = TRUE)
r <- (log2(x/mean(x)))

mydata <- list(r = r, f = f, N=length(r), s = l, Nsub = Nsub, psi = mean(x)/100)

code <- '
data {
  int N;
  real r[N];
  real f[N];
  int<lower=1> Nsub;
  int<lower=1> s[N];
}
parameters {
  real<lower=0.1, upper=10> cn[Nsub];
  real<lower=0.1, upper=cn> m[Nsub];
  vector<lower=0>[Nsub] sigma_cn;
  vector<lower=0>[Nsub] sigma_m;
  real<lower=0, upper=1.0> P;
}
transformed parameters {
  //real alpha[Nsub];
  //real m[Nsub];
  real psi;
  //for (i in 1:Nsub){
  //alpha = cn/2;
  //}
  //for (i in 1:Nsub){
  //m[i]<lower=0, upper=alpha[i]> = m[i];
  //}
  psi = mean(cn);
}
model {
  for(i in 1:N){
  r[i] ~ normal(log2((P*cn[s[i]] + 2*(1-P))/(P*psi + 2*(1-P))),sigma_cn[s[i]]);
  f[i] ~ normal((P*m[s[i]]+1-P)/(P*cn[s[i]]+2*(1-P)), sigma_m[s[i]]);
  }
}
'

fit <- stan(model_code = code, data = mydata, iter = 100, 
            chains = 2, control = list(adapt_delta = 0.99,
                                       max_treedepth = 15))

plot(fit, pars = 'cn')
plot(fit, pars = 'm')

traceplot(fit, pars = 'cn')

stan_diag(fit)
post <- extract(fit)

hist(post$P,
     main = paste("Tumor purity"),
     ylab = '')

