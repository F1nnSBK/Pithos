import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const config: Config = {
  title: 'Pithos',
  tagline: 'Model-Isomorphic Database for massive Datasets',
  favicon: 'img/favicon.ico',

  customFields: {
    version: process.env.PITHOS_VERSION || 'dev',
  },

  url: 'https://F1nnSBK.github.io',
  baseUrl: '/lcvk/',

  organizationName: 'F1nnSBK',
  projectName: 'lcvk',

  onBrokenLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/F1nnSBK/lcvk/tree/develop-docs/website/',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
      type: 'text/css',
      integrity:
        'sha384-odtC+0UGzzFDQ+9G/8W0bZgPqL2Pq+N7/pP81RjF/w5Wk+q4v6Oa+zJ7rT3XQG2+',
      crossorigin: 'anonymous',
    },
    {
      href: 'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@400;600;700;800&display=swap',
      type: 'text/css',
      crossorigin: 'anonymous',
    },
  ],

  themes: ['@docusaurus/theme-mermaid'],
  markdown: {
    mermaid: true,
  },
  themeConfig: {
    colorMode: {
      defaultMode: 'dark',
      disableSwitch: true,
      respectPrefersColorScheme: false,
    },
    // Replace with your project's social card
    image: 'img/docusaurus-social-card.jpg',
    navbar: {
      title: '[ PITHOS ]',
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Documentation',
        },
        {
          href: 'https://github.com/F1nnSBK/lcvk',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {
              label: 'Getting Started',
              to: '/docs/ARCHITECTURAL_PRINCIPLES',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/F1nnSBK/lcvk',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Pithos. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.dracula,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
