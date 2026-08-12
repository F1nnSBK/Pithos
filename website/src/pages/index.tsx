import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero glass-header', styles.heroBanner)} style={{ position: 'relative', overflow: 'hidden' }}>
      <div className="container" style={{ position: 'relative', zIndex: 1 }}>
        <div style={{ fontSize: '5rem', marginBottom: '1rem', color: 'var(--ifm-color-primary)' }}>
          ⚱
        </div>
        <Heading as="h1" className="hero__title hero-gradient-text" style={{ fontSize: '4rem', marginBottom: '0.5rem' }}>
          {siteConfig.title}
        </Heading>
        <div style={{ color: 'var(--ifm-color-primary-dark)', fontFamily: 'Outfit, sans-serif', fontSize: '1.2rem', marginBottom: '1rem', letterSpacing: '2px' }}>
          v{siteConfig.customFields.version as string}
        </div>
        <p className="hero__subtitle" style={{ color: '#aaa', fontWeight: 500, fontSize: '1.5rem' }}>
          {siteConfig.tagline}
        </p>
        <div className={styles.buttons} style={{ marginTop: '3rem', display: 'flex', gap: '1.5rem', justifyContent: 'center', flexWrap: 'wrap' }}>
          <Link
            className="button button--primary button--lg premium-button"
            to="/docs/ARCHITECTURAL_PRINCIPLES">
            Architectural Principles
          </Link>
          <Link
            className="button button--outline button--primary button--lg premium-button"
            to="/docs/C_API_REFERENCE">
            C-API Reference
          </Link>
          <Link
            className="button button--outline button--primary button--lg premium-button"
            href="https://github.com/F1nnSBK/lcvk/releases/latest">
            Download Latest Release
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="Model-Isomorphic Database for massive Datasets">
      <HomepageHeader />
      <main style={{ padding: '5rem 2rem', textAlign: 'center', backgroundColor: '#050505' }}>
        <Heading as="h2" className="hero-gradient-text" style={{ fontSize: '2.5rem', marginBottom: '2rem' }}>Why Pithos?</Heading>
        <p style={{ maxWidth: '800px', margin: '0 auto', color: '#ccc', fontSize: '1.25rem', lineHeight: '1.8', fontFamily: 'Inter, sans-serif' }}>
          Pithos minimizes latency for massive datasets by bypassing traditional language runtimes and garbage collection. 
          It maps index structures directly to off-heap virtual memory using POSIX-aligned files, enabling hardware-level execution speeds.
        </p>
      </main>
    </Layout>
  );
}
