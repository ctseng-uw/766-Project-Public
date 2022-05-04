import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  imgSrc: string;
  description: JSX.Element;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Motivation',
    imgSrc: require('@site/static/img/yen.jpeg').default,
    description: (
      <>
        It will be cool to have a personalized handwriting font, and since there are more than 80,000 characters in Chinese, it is not efficient to write down each character to create a font. We would like to find the way that can utilized limited characters to transfer the style to other characters.
      </>
    ),
  },
  {
    title: 'Approach',
    imgSrc: require('@site/static/img/result.gif').default,
    description: (
      <>
        We consider the method of style transfer, and implement different GANs that are notable in the area. Then, we modified the existing approach to improve the performance.
      </>
    ),
  },
  {
    title: 'Preivew of Our Result',
    imgSrc: require('@site/static/img/preview.png').default,
    description: (
      <>
        Our approach is good at handwriting-style font, which is an improvement from the original version. There are more results with different fonts in our report.
      </>
    ),
  },
];

function Feature({ title, imgSrc, description }: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <img src={imgSrc} className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): JSX.Element {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
