# 最小样本量的推导



![image-20250906000148556](/Users/kimzhao/Documents/gongzhonghao/PublicArticle/ABT/Chapter14.assets/image-20250906000148556.png)

根据统计功效定义有$P(拒绝原假设H_0|原假设H_0错误)=1-\beta$

以上公式暗含两个问题：

1. 在原假设下，拒绝原假设需要样本均值的差值满足什么条件。

   以双边检验为例，我们知道，当样本差值满足$P(\frac{\bar{x}_1-\bar{x}_2-0}{\sqrt{\frac{\sigma^2_1}{n_1}+\frac{\sigma^2_1}{n_2}}}>=Z_{\frac{\alpha}{2}})=\frac{\alpha}{2}$时，我们会拒绝原假设。其中$Z_{\frac{\alpha}{2}}$为标准正态分布上$\frac{\alpha}{2}$分位数。解上式有$\bar{x}_1-\bar{x}_2>=Z_{\frac{\alpha}{2}}\times\sqrt{\frac{\sigma^2_1}{n_1}+\frac{\sigma^2_2}{n_2}}$

2. 在备择假设正确的条件下，且统计功效等于$1-\beta$时, 我们有
   $$
   \begin{align}
   P(\bar{x}_1-\bar{x}_2>=Z_{\frac{\alpha}{2}}\times\sqrt{\frac{\sigma^2_1}{n_1}+\frac{\sigma^2_2}{n_2}})&=1-\beta
   \\基于H_1假设，标准化后有
   P(\frac{\bar{x}_1-\bar{x}_2-\Delta}{\sqrt{\frac{\sigma_1^2}{n_1}+\frac{\sigma_2^2}{n_2}}}>=Z_{\frac{\alpha}{2}}-\frac{\Delta}{\sqrt{\frac{\sigma_1^2}{n_1}+\frac{\sigma_2^2}{n_2}}})&=1-\beta\\
   基于标准正态分布的定义有，
   Z_{\frac{\alpha}{2}}-\frac{\Delta}{\sqrt{\frac{\sigma_1^2}{n_1}+\frac{\sigma_2^2}{n_2}}}&=Z_{1-\beta}\\
   移项有，
   \sqrt{\frac{\sigma_1^2}{n_1}+\frac{\sigma_2^2}{n_2}}&=\frac{\Delta}{Z_{\frac{\alpha}{2}}-Z_{1-\beta}}\\
   等式两边同时平方有,
   \frac{\sigma^2}{n_1}+\frac{\sigma^2}{n_2}&=\frac{\Delta^2}{(Z_{\frac{\alpha}{2}}-Z_{1-\beta})^2}\\
   当n_1=n_2=\frac{n}{2}时有，\frac{\sigma_1^2+\sigma_2^2}{n_1} &= \frac{\Delta^2}{(Z_{\frac{\alpha}{2}}-Z_{1-\beta})^2}\\
   移项有，n_1&=\frac{(Z_{\frac{\alpha}{2}}-Z_{1-\beta})^2(\sigma_1^2+\sigma_2^2)}{\Delta^2}\\
   根据正态分布的对称性有，Z_{\frac{\alpha}{2}}=-Z_{1-\frac{\alpha}{2}},带入有，n_1&=\frac{(Z_{1-\frac{\alpha}{2}}+Z_{1-\beta})^2(\sigma_1^2+\sigma_2^2)}{\Delta^2}\\
   当\sigma^2_1=\sigma^2_2=\sigma^2时，有n_1&=\frac{(Z_{1-\frac{\alpha}{2}}+Z_{1-\beta})^2\times2\sigma^2}{\Delta^2}
   \end{align}
   $$