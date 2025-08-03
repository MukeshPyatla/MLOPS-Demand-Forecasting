# 🚀 Streamlit Cloud Deployment Guide

This guide will help you deploy your MLOps Demand Forecasting application to Streamlit Cloud.

## Prerequisites

1. **GitHub Account**: Make sure your code is pushed to a GitHub repository
2. **Streamlit Cloud Account**: Sign up at [share.streamlit.io](https://share.streamlit.io)

## Deployment Steps

### 1. Prepare Your Repository

Ensure your repository has the following files:
- `streamlit_app.py` (main Streamlit application)
- `requirements.txt` (Python dependencies)
- `.streamlit/config.toml` (Streamlit configuration)
- `packages.txt` (system dependencies)

### 2. Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**: Visit [share.streamlit.io](https://share.streamlit.io)
2. **Sign in with GitHub**: Connect your GitHub account
3. **New App**: Click "New app"
4. **Configure Deployment**:
   - **Repository**: Select your GitHub repository
   - **Branch**: Choose `main` (or your default branch)
   - **Main file path**: Enter `streamlit_app.py`
   - **App URL**: Choose a custom URL (optional)

### 3. Advanced Configuration

#### Environment Variables (if needed)
If your app requires environment variables (like API keys), you can add them in the Streamlit Cloud dashboard:
1. Go to your app settings
2. Navigate to "Secrets"
3. Add your environment variables

#### Custom Domain (Optional)
You can configure a custom domain for your app:
1. Go to app settings
2. Navigate to "Custom domain"
3. Follow the DNS configuration instructions

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are in `requirements.txt`
2. **System Dependencies**: Add required system packages to `packages.txt`
3. **Memory Issues**: Optimize your app for memory usage
4. **Timeout Errors**: Ensure your app loads within the timeout limit

### Performance Tips

1. **Caching**: Use `@st.cache_data` for expensive computations
2. **Lazy Loading**: Load data only when needed
3. **Optimize Imports**: Import heavy libraries only when required
4. **Reduce Dependencies**: Keep requirements.txt minimal

## Monitoring

- **Logs**: View app logs in the Streamlit Cloud dashboard
- **Analytics**: Monitor app usage and performance
- **Alerts**: Set up notifications for errors

## Security Best Practices

1. **Secrets Management**: Use Streamlit secrets for sensitive data
2. **Input Validation**: Validate all user inputs
3. **Rate Limiting**: Implement rate limiting for API calls
4. **HTTPS**: All Streamlit Cloud apps use HTTPS by default

## Support

- **Documentation**: [docs.streamlit.io](https://docs.streamlit.io)
- **Community**: [discuss.streamlit.io](https://discuss.streamlit.io)
- **GitHub Issues**: Report bugs on the Streamlit GitHub repository

## Example Configuration

Your repository structure should look like this:
```
your-repo/
├── streamlit_app.py
├── requirements.txt
├── packages.txt
├── .streamlit/
│   └── config.toml
├── src/
│   ├── data/
│   ├── processing/
│   └── training/
└── README.md
```

## Quick Deploy Checklist

- [ ] Code pushed to GitHub
- [ ] `streamlit_app.py` exists and runs locally
- [ ] `requirements.txt` includes all dependencies
- [ ] `.streamlit/config.toml` configured
- [ ] `packages.txt` includes system dependencies (if needed)
- [ ] App tested locally with `streamlit run streamlit_app.py`
- [ ] No hardcoded secrets in code
- [ ] All imports working correctly 