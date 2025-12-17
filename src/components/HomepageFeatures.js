import React from 'react';
import clsx from 'clsx';
import styles from './HomepageFeatures.module.css';

const FeatureList = [
  {
    title: 'Physical AI Focus',
    description: (
      <>
        Learn how to create AI systems that interact with the physical world,
        bridging the gap between digital intelligence and real-world applications.
      </>
    ),
  },
  {
    title: 'Hands-On Projects',
    description: (
      <>
        Build practical projects from line-following robots to autonomous navigation systems.
        Learn by doing with real code examples and exercises.
      </>
    ),
  },
  {
    title: 'Safety & Ethics First',
    description: (
      <>
        Understand critical safety and ethical considerations in physical AI systems.
        Learn to build responsible and trustworthy robots.
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
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